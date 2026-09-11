"""
REST API: POST /search, POST /check_duplicate, POST /deduplicate_batch, GET /health.
Swagger на /docs. Инференс на CPU.
"""

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config as cfg
import matcher
import jellyfish

from normalize import phonetic_key, sorted_tokens

app = FastAPI(
    title="ClearPic Entity Resolution API",
    description=(
        "Поиск и дедупликация записей о физических и юридических лицах "
        "с поддержкой мультиязычности и нечёткого сравнения."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# состояние сервиса
# ---------------------------------------------------------------------------

STATE = {"model": None, "names": [], "index": {}}


def build_index(names):
    """
    Инвертированный индекс по ключам блокинга.

    Без него поиск потребовал бы сравнения запроса со всей базой. С ним
    мы достаём только записи, попавшие хотя бы в один общий блок с запросом,
    и уже их отдаём модели.
    """
    index = {}
    for i, lat in enumerate(names):
        for k in _keys(lat):
            index.setdefault(k, []).append(i)
    return index


def _keys(lat: str):
    toks = lat.split()
    if not toks:
        return []
    keys = [
        ("sorted", sorted_tokens(lat)),
        ("phon", phonetic_key(lat)),
        # Soundex: в поиске повышает полноту кандидатов (0,489 -> 0,598, ноутбук 03);
        # в пакетной дедупликации отключён - раздувает кластеры через транзитивное замыкание.
        ("sdx", " ".join(sorted(jellyfish.soundex(t) for t in toks if t))),
    ]
    if len(toks) >= 2:
        a, b = sorted(toks)[:2]
        keys.append(("pref", a[:4] + "|" + b[:4]))
    return keys


@app.on_event("startup")
def startup():
    if os.path.exists(matcher.MODEL_PATH):
        STATE["model"] = matcher.load()

    # Индекс для /search строится из parquet с именами; путь задаётся
    # переменной окружения ER_INDEX. Остальные ручки работают и без индекса.
    path = os.environ.get("ER_INDEX")
    if path and os.path.exists(path):
        import pandas as pd
        df = pd.read_parquet(path)
        col = cfg.COL_NAME if cfg.COL_NAME in df.columns else df.columns[0]
        raw = df[col].astype(str).drop_duplicates().tolist()
        STATE["names"] = [matcher.prep(x) for x in raw]
        STATE["raw"] = raw
        STATE["index"] = build_index(STATE["names"])


# ---------------------------------------------------------------------------
# схемы запросов и ответов
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    name: str = Field(..., description="Имя или название для поиска", examples=["Ivanov Petr"])
    top_k: int = Field(10, ge=1, le=100, description="Сколько результатов вернуть")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Порог уверенности")


class Match(BaseModel):
    name: str
    score: float = Field(..., description="Вероятность того, что это одна сущность")


class SearchResponse(BaseModel):
    query: str
    normalized: str = Field(..., description="Запрос после нормализации и транслитерации")
    candidates_examined: int
    results: List[Match]


class PairRequest(BaseModel):
    left: str = Field(..., examples=["Иванов Пётр Сергеевич"])
    right: str = Field(..., examples=["Ivanoff Petr"])


class PairResponse(BaseModel):
    left_normalized: str
    right_normalized: str
    probability: float
    is_duplicate: bool
    threshold: float


class BatchRequest(BaseModel):
    names: List[str] = Field(..., max_length=5000)
    threshold: float = Field(0.9, ge=0.0, le=1.0)


class BatchResponse(BaseModel):
    n_records: int
    n_clusters: int
    cluster_ids: List[int]


# ---------------------------------------------------------------------------
# ручки
# ---------------------------------------------------------------------------


@app.get("/health", summary="Проверка готовности")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "index_size": len(STATE["names"]),
    }


@app.post("/check_duplicate", response_model=PairResponse,
          summary="Вероятность того, что две записи - дубликаты")
def check_duplicate(req: PairRequest, threshold: float = 0.9):
    if STATE["model"] is None:
        raise HTTPException(503, "модель не загружена")
    a, b = matcher.prep(req.left), matcher.prep(req.right)
    p = float(matcher.predict_pairs(STATE["model"], [a], [b])[0])
    return PairResponse(left_normalized=a, right_normalized=b, probability=p,
                        is_duplicate=p >= threshold, threshold=threshold)


@app.post("/search", response_model=SearchResponse,
          summary="Top-K похожих записей из базы")
def search(req: SearchRequest):
    if STATE["model"] is None:
        raise HTTPException(503, "модель не загружена")
    if not STATE["names"]:
        raise HTTPException(503, "индекс пуст: задайте переменную окружения ER_INDEX")

    q = matcher.prep(req.name)

    # блокинг: берём только записи, делящие с запросом хотя бы один ключ
    cand = set()
    for k in _keys(q):
        cand.update(STATE["index"].get(k, []))
    cand = list(cand)
    if not cand:
        return SearchResponse(query=req.name, normalized=q,
                              candidates_examined=0, results=[])

    names = [STATE["names"][i] for i in cand]
    proba = matcher.predict_pairs(STATE["model"], [q] * len(names), names)

    order = np.argsort(-proba)[: req.top_k]
    results = [Match(name=STATE["raw"][cand[i]], score=float(proba[i]))
               for i in order if proba[i] >= req.min_score]
    return SearchResponse(query=req.name, normalized=q,
                          candidates_examined=len(cand), results=results)


@app.post("/deduplicate_batch", response_model=BatchResponse,
          summary="Кластеризация батча записей")
def deduplicate_batch(req: BatchRequest):
    if STATE["model"] is None:
        raise HTTPException(503, "модель не загружена")

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    lat = [matcher.prep(x) for x in req.names]
    n = len(lat)

    # блокинг внутри батча
    blocks = {}
    for i, s in enumerate(lat):
        for k in _keys(s):
            blocks.setdefault(k, []).append(i)

    pairs = set()
    for idx in blocks.values():
        if 2 <= len(idx) <= 200:
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    x, y = idx[a], idx[b]
                    pairs.add((x, y) if x < y else (y, x))

    if pairs:
        arr = np.array(sorted(pairs))
        proba = matcher.predict_pairs(STATE["model"],
                                      [lat[i] for i in arr[:, 0]],
                                      [lat[j] for j in arr[:, 1]])
        e = arr[proba >= req.threshold]
    else:
        e = np.empty((0, 2), dtype=int)

    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)
    return BatchResponse(n_records=n, n_clusters=int(n_comp),
                         cluster_ids=[int(x) for x in labels])
