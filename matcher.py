"""Признаки пары имён, обучение модели, предсказание."""

import os
import pickle
import random

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

import config as cfg
from augment import corrupt
from normalize import normalize, phonetic_key, translit

# ---------------------------------------------------------------------------
# признаки
# ---------------------------------------------------------------------------

# Каждая мера ловит свой тип расхождения. По отдельности ни одна не отличает
# однофамильцев от настоящих дублей - поэтому нужна модель поверх них.
SIM_FUNCS = {
    "ratio": lambda a, b: fuzz.ratio(a, b) / 100,
    "token_sort": lambda a, b: fuzz.token_sort_ratio(a, b) / 100,
    "token_set": lambda a, b: fuzz.token_set_ratio(a, b) / 100,
    "partial": lambda a, b: fuzz.partial_ratio(a, b) / 100,
    "jaro_winkler": lambda a, b: distance.JaroWinkler.similarity(a, b),
    "levenshtein": lambda a, b: 1 - distance.Levenshtein.normalized_distance(a, b),
}

FEATURE_NAMES = list(SIM_FUNCS) + [
    "len_diff", "tok_diff", "same_phon", "same_init", "min_tokens",
]


def pair_features(left, right) -> pd.DataFrame:
    """
    Признаки для набора пар. На вход - уже нормализованные латинские строки.

    Возвращает DataFrame с колонками FEATURE_NAMES в фиксированном порядке:
    порядок важен, потому что модель обучалась именно на нём.
    """
    left = [str(x) for x in left]
    right = [str(x) for x in right]

    data = {name: [f(a, b) for a, b in zip(left, right)] for name, f in SIM_FUNCS.items()}

    ll = np.array([len(a) for a in left], dtype=float)
    rl = np.array([len(b) for b in right], dtype=float)
    lt = np.array([len(a.split()) for a in left], dtype=float)
    rt = np.array([len(b.split()) for b in right], dtype=float)

    data["len_diff"] = np.abs(ll - rl) / (ll + rl + 1)
    data["tok_diff"] = np.abs(lt - rt)
    data["same_phon"] = [int(phonetic_key(a) == phonetic_key(b)) for a, b in zip(left, right)]
    data["same_init"] = [
        int("".join(t[0] for t in a.split() if t) == "".join(t[0] for t in b.split() if t))
        for a, b in zip(left, right)
    ]
    data["min_tokens"] = np.minimum(lt, rt)

    return pd.DataFrame(data, columns=FEATURE_NAMES).astype(float)


def prep(s: str) -> str:
    """Единая точка приведения строки к сравнимому виду."""
    return translit(normalize(s))


# ---------------------------------------------------------------------------
# обучающий набор
# ---------------------------------------------------------------------------


def make_training_pairs(names, n_pos=60_000, n_neg_hard=30_000, n_neg_easy=15_000,
                        seed=cfg.RANDOM_STATE, verbose=True):
    """
    Строит обучающий набор из списка реальных имён.

    Позитивы - синтетические искажения (gold-пары для этого непригодны:
    внутри gold-кластера 100% написаний идентичны).
    Негативы - однофамильцы (трудные) плюс случайные пары (лёгкие).
    """
    import collections
    from rapidfuzz import process

    rng = random.Random(seed)
    say = print if verbose else (lambda *a, **k: None)

    # --- позитивы --------------------------------------------------------
    say("генерирую позитивные пары...")
    base = rng.sample(list(names), min(n_pos, len(names)))
    pos = []
    for name in base:
        n_ops = rng.choices([1, 2, 3], weights=[5, 3, 2], k=1)[0]
        right, ops = corrupt(name, n_ops=n_ops, seed=rng.randrange(10**9))
        if right and right != name:
            pos.append((prep(name), prep(right), 1, n_ops, "синтетический дубль"))

    # --- трудные негативы: однофамильцы ----------------------------------
    say("ищу однофамильцев...")
    lat = sorted({prep(x) for x in names})
    lat = [x for x in lat if len(x.split()) >= 2]

    by_surname = collections.defaultdict(list)
    for s in lat:
        by_surname[s.split()[0]].append(s)

    hard = []
    for group in by_surname.values():
        if len(group) < 2:
            continue
        for name in group[:8]:
            others = [g for g in group if g != name]
            best = process.extractOne(name, others, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= 60:
                hard.append((name, best[0], 0, 0, "однофамильцы"))
        if len(hard) >= n_neg_hard:
            break
    hard = hard[:n_neg_hard]

    # --- лёгкие негативы: случайные пары ---------------------------------
    easy = [(rng.choice(lat), rng.choice(lat), 0, 0, "случайные")
            for _ in range(n_neg_easy)]

    pairs = pd.DataFrame(pos + hard + easy,
                         columns=["left", "right", "label", "n_ops", "kind"])
    say(f"пар: {len(pairs):,}".replace(",", " "))
    return pairs


def split_by_entity(pairs, seed=cfg.RANDOM_STATE):
    """
    Сплит по сущностям, а не по парам.

    Если делить пары случайно, записи одной сущности попадут и в train,
    и в test - модель запомнит имя, и метрики окажутся завышены утечкой.
    """
    from sklearn.model_selection import train_test_split

    groups = np.asarray(pairs["left"].astype(object).unique(), dtype=object)
    tr_g, rest = train_test_split(groups, test_size=0.3, random_state=seed)
    va_g, te_g = train_test_split(rest, test_size=0.5, random_state=seed)

    s = pd.Series("train", index=pairs.index)
    s[pairs["left"].isin(va_g)] = "valid"
    s[pairs["left"].isin(te_g)] = "test"
    return s.values


# ---------------------------------------------------------------------------
# модель
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(cfg.WORK, "matcher.pkl")


def train(pairs, verbose=True):
    """Градиентный бустинг поверх признаков схожести."""
    import lightgbm as lgb

    X = pair_features(pairs["left"], pairs["right"])
    y = pairs["label"].to_numpy(dtype=int)
    sp = pairs["split"].values

    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                               num_leaves=31, random_state=cfg.RANDOM_STATE,
                               verbose=-1)
    model.fit(X[sp == "train"], y[sp == "train"],
              eval_set=[(X[sp == "valid"], y[sp == "valid"])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    if verbose:
        print("обучено, деревьев:", model.n_estimators_)
    return model, X, y


def save(model, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_NAMES}, f)
    return path


def load(path=MODEL_PATH):
    with open(path, "rb") as f:
        d = pickle.load(f)
    assert d["features"] == FEATURE_NAMES, "набор признаков изменился - переобучите модель"
    return d["model"]


def predict_pairs(model, left, right, batch=200_000):
    """Вероятность дубля для набора пар. Батчами, чтобы не съесть память."""
    out = []
    for i in range(0, len(left), batch):
        X = pair_features(left[i:i + batch], right[i:i + batch])
        out.append(model.predict_proba(X)[:, 1])
    return np.concatenate(out) if out else np.array([])
