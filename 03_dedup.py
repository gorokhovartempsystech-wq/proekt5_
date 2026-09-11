"""
Дедупликация всей базы: блокинг, скоринг, кластеризация.
Результат: work/deduplicated.parquet с cluster_id для каждой записи, work/dedup_report.txt.
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config as cfg
import matcher
import jellyfish

from normalize import phonetic_key, sorted_tokens

# ---------------------------------------------------------------------------
# блокинг
# ---------------------------------------------------------------------------

MAX_BLOCK = 60          # блоки больше этого пропускаем: они дают квадратичный взрыв
                        # при нулевой пользе (это обычно служебные и пустые значения


def blocking_keys(lat: str, use_soundex: bool = False):
    """
    Ключи блокинга; записи сравниваются, если совпал хотя бы один.

    sorted - токены по алфавиту (порядок ФИО); phon - metaphone (латинизация);
    pref - префиксы двух первых токенов (опечатки в хвосте); sdx - Soundex,
    по умолчанию выключен: полнее, но на полной базе раздувает кластеры
    (крупнейший кластер 14 521 -> 20 971, cross-country 19 906 -> 21 639). Включён только в поиске.
    """
    toks = lat.split()
    if not toks:
        return []
    keys = [
        ("sorted", sorted_tokens(lat)),
        ("phon", phonetic_key(lat)),
    ]
    if len(toks) >= 2:
        a, b = sorted(toks)[:2]
        keys.append(("pref", a[:4] + "|" + b[:4]))
    if use_soundex:
        keys.append(("sdx", " ".join(sorted(jellyfish.soundex(t) for t in toks if t))))
    return keys


def build_candidates(lat_values, max_block=MAX_BLOCK, verbose=True, use_soundex=False):
    """
    Возвращает массив пар индексов-кандидатов (уникальные, i < j).
    """
    say = print if verbose else (lambda *a, **k: None)
    blocks = defaultdict(list)

    t0 = time.time()
    for i, lat in enumerate(lat_values):
        for k in blocking_keys(lat, use_soundex):
            blocks[k].append(i)
        if verbose and i and i % 500_000 == 0:
            say(f"    индексирую {i:,}".replace(",", " "))
    say(f"  блоков: {len(blocks):,}".replace(",", " ") + f"   ({time.time()-t0:.0f} с)")

    pairs = set()
    skipped = 0
    for key, idx in blocks.items():
        n = len(idx)
        if n < 2:
            continue
        if n > max_block:
            skipped += 1
            continue
        for a in range(n):
            for b in range(a + 1, n):
                x, y = idx[a], idx[b]
                pairs.add((x, y) if x < y else (y, x))

    say(f"  пропущено слишком больших блоков: {skipped:,}".replace(",", " "))
    say(f"  пар-кандидатов: {len(pairs):,}".replace(",", " "))
    return np.array(sorted(pairs), dtype=np.int64) if pairs else np.empty((0, 2), np.int64)


# ---------------------------------------------------------------------------
# основной сценарий
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="порог склейки; 0 = выбрать автоматически по контрольным числам")
    ap.add_argument("--sample", type=int, default=0,
                    help="ограничить N записями (для отладки)")
    ap.add_argument("--max-block", type=int, default=MAX_BLOCK)
    ap.add_argument("--soundex", action="store_true",
                    help="добавить Soundex в блокинг (полнее, но переклеивает)")
    ap.add_argument("--stop-freq", type=int, default=50,
                    help="имя у стольких и более разных компаний считается плейсхолдером")
    args = ap.parse_args()

    lines = []

    def rep(*p):
        s = " ".join(str(x) for x in p)
        print(s, flush=True)
        lines.append(s)

    def human(n):
        return f"{n:,}".replace(",", " ")

    rep("=" * 70)
    rep("ДЕДУПЛИКАЦИЯ БАЗЫ")
    rep("=" * 70)

    # --- данные ----------------------------------------------------------
    df = pd.read_parquet(cfg.DATA, columns=[cfg.COL_ID, cfg.COL_COUNTRY,
                                            cfg.COL_PARTY_TYPE, cfg.COL_NAME,
                                            cfg.COL_GOLD, cfg.COL_COMPANY_ID])
    if args.sample:
        df = df.sample(args.sample, random_state=cfg.RANDOM_STATE).reset_index(drop=True)
    rep(f"записей: {human(len(df))}")

    # --- нормализация ----------------------------------------------------
    rep("")
    rep("1. НОРМАЛИЗАЦИЯ")
    t0 = time.time()
    raw = df[cfg.COL_NAME].astype(str)
    cache = {v: matcher.prep(v) for v in raw.unique()}
    lat = raw.map(cache).astype(object)
    df["lat"] = lat.values
    rep(f"  уникальных имён: {human(raw.nunique())}")
    rep(f"  после нормализации: {human(lat.nunique())}")
    rep(f"  время: {time.time()-t0:.0f} с")

    # --- модель ----------------------------------------------------------
    rep("")
    rep("2. МОДЕЛЬ")
    if os.path.exists(matcher.MODEL_PATH):
        model = matcher.load()
        rep("  загружена из work/matcher.pkl")
    else:
        # та же выборка и тот же seed, что в 04_quality_report.py:
        # модель в выгрузке и модель в отчёте о качестве должны совпадать
        rep("  обучаю заново...")
        names = (raw[df[cfg.COL_PARTY_TYPE] == "individual"]
                 .sample(min(300_000, len(raw)), random_state=cfg.RANDOM_STATE).tolist())
        pairs = matcher.make_training_pairs(names)
        pairs["split"] = matcher.split_by_entity(pairs)
        model, _, _ = matcher.train(pairs)
        matcher.save(model)
        rep("  сохранена в work/matcher.pkl")

    # --- стоп-имена ------------------------------------------------------
    # Плейсхолдеры («анонимный акционер», «неизвестная организация») - не сущности;
    # без отсечения 50 тыс. записей схлопываются в один кластер.
    # Частоту считаем по числу РАЗНЫХ компаний, а не строк: плейсхолдер стоит
    # у тысяч компаний, реальный человек - у нескольких, даже если источник
    # даёт десятки снапшотов на него (в Армении медиана 40 строк на имя).
    # Правило по строкам ошибочно отсекало 83% армянских записей.
    rep("")
    rep("3. СТОП-ЛИСТ ПО ЧИСЛУ КОМПАНИЙ")
    n_comp = df.groupby("lat")[cfg.COL_COMPANY_ID].nunique()
    stop = set(n_comp[n_comp >= args.stop_freq].index)
    # имена короче 3 символов после нормализации (пусто, "а", "--") тоже не идентификатор
    too_short = set(x for x in df["lat"].unique() if len(x) < 3)
    stop |= too_short
    is_stop = df["lat"].isin(stop)
    rep(f"  порог: имя встречается у >= {args.stop_freq} разных компаний")
    rep(f"  имён в стоп-листе:        {human(len(stop))}  (из них короче 3 символов: {len(too_short)})")
    rep(f"  записей под ними:         {human(int(is_stop.sum()))}  ({is_stop.mean():.1%} базы)")
    rep("  Эти записи не резолвятся: имя не является идентификатором.")
    rep("  В выгрузке они получают cluster_id = -1.")

    # --- блокинг ---------------------------------------------------------
    rep("")
    rep("4. БЛОКИНГ")
    uniq = pd.Index([x for x in df["lat"].unique() if x not in stop])
    rep(f"  сравниваем уникальные имена: {human(len(uniq))}")
    rep(f"  полный перебор потребовал бы: {human(len(uniq)*(len(uniq)-1)//2)} пар")

    t0 = time.time()
    cand = build_candidates(uniq.tolist(), max_block=args.max_block,
                            use_soundex=args.soundex)
    rep(f"  пар-кандидатов: {human(len(cand))}")
    rep(f"  время блокинга: {time.time()-t0:.0f} с")
    if len(cand):
        red = 1 - len(cand) / (len(uniq) * (len(uniq) - 1) / 2)
        rep(f"  сокращение перебора: {red:.6%}")

    # --- скоринг ---------------------------------------------------------
    rep("")
    rep("5. СКОРИНГ")
    t0 = time.time()
    left = uniq[cand[:, 0]].tolist()
    right = uniq[cand[:, 1]].tolist()
    proba = matcher.predict_pairs(model, left, right)
    rep(f"  оценено пар: {human(len(proba))}   ({time.time()-t0:.0f} с)")
    for t in (0.5, 0.7, 0.9):
        rep(f"    порог {t}: рёбер {human(int((proba >= t).sum()))}")

    # --- калибровка порога ------------------------------------------------
    rep("")
    rep("6. КАЛИБРОВКА ПОРОГА ПО КОНТРОЛЬНЫМ ЧИСЛАМ")
    rep("")
    rep("  Контрольные числа: 3693 персоны и 182 компании в 2+ странах.")
    rep("  Меток нет; по числу подбирается порог.")
    rep("")
    TARGET = 3693 + 182
    n = len(uniq)
    grid = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99]
    rows = []
    best = None
    for t in grid:
        e = cand[proba >= t]
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        s_map = pd.Series(lab, index=uniq)
        cl = df["lat"].map(s_map)
        cl = cl.where(~is_stop.values, -1)
        sz = cl[cl >= 0].value_counts()
        dup_share = sz[sz > 1].sum() / len(df)
        biggest = int(sz.max())
        pc = df.assign(_c=cl.values).groupby("_c")[cfg.COL_COUNTRY].nunique()
        xc = int((pc > 1).sum())
        rows.append((t, len(e), int(sz.shape[0]), dup_share, biggest, xc))
        rep(f"  порог {t:<5} рёбер {human(len(e)):>10}  кластеров {human(int(sz.shape[0])):>10}"
            f"  дублей {dup_share:>6.1%}  макс {human(biggest):>8}  cross-country {human(xc):>7}")
        if best is None or abs(xc - TARGET) < abs(best[1] - TARGET):
            best = (t, xc)

    rep("")
    rep(f"  ближе всего к контрольному числу {TARGET}: порог {best[0]} ({human(best[1])})")

    chosen = args.threshold if args.threshold > 0 else best[0]
    rep("")
    rep(f"7. ИТОГОВАЯ КЛАСТЕРИЗАЦИЯ (порог {chosen})")
    e = cand[proba >= chosen]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)
    cl = df["lat"].map(pd.Series(labels, index=uniq))
    df["cluster_id"] = cl.where(~is_stop.values, -1).fillna(-1).astype("int64")

    resolved = df[df.cluster_id >= 0]
    sizes = resolved.groupby("cluster_id").size()
    rep(f"  не резолвится (стоп-лист): {human(int((df.cluster_id < 0).sum()))}")
    rep(f"  кластеров:              {human(int(sizes.shape[0]))}")
    rep(f"  записей в кластерах 2+: {human(int(sizes[sizes > 1].sum()))}")
    rep(f"  доля дублей:            {sizes[sizes > 1].sum() / len(resolved):.1%}")
    rep(f"  крупнейший кластер:     {human(int(sizes.max()))}")
    for q in (0.5, 0.9, 0.99, 0.999):
        rep(f"    p{q*100:<6g} {int(sizes.quantile(q))}")

    per_country = resolved.groupby("cluster_id")[cfg.COL_COUNTRY].nunique()
    rep(f"  кластеров в 2+ странах: {human(int((per_country > 1).sum()))}   "
        f"(контроль {TARGET})")

    rep("")
    rep("8. ПРОВЕРКА НА GOLD-РАЗМЕТКЕ")
    gold = df[df[cfg.COL_GOLD].notna()]
    if len(gold):
        same_gold = gold.groupby(cfg.COL_GOLD)["cluster_id"].nunique()
        rep(f"  сущностей в разметке:   {human(len(same_gold))}")
        rep(f"  собраны в один кластер: {(same_gold == 1).mean():.2%}")
        rep("  (recall на внутристрановых дублях)")

    # --- выгрузка --------------------------------------------------------
    out = os.path.join(cfg.WORK, "deduplicated.parquet")
    df[[cfg.COL_ID, cfg.COL_COUNTRY, cfg.COL_PARTY_TYPE, "cluster_id"]].to_parquet(
        out, index=False)
    rep("")
    rep(f"выгрузка: {os.path.relpath(out)}")

    rp = os.path.join(cfg.WORK, "dedup_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    rep(f"отчёт:    {os.path.relpath(rp)}")


if __name__ == "__main__":
    main()
