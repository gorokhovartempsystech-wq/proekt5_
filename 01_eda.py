"""
Разведочный анализ датасета: структура, пропуски, алфавиты, gold-разметка, размеры кластеров.
Значения показываются только для колонок с малой мощностью; флаг --no-values отключает и их.

    python 01_eda.py путь.parquet --gold party_public_id
"""

import argparse
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# определение алфавита
# ----------------------------------------------------------------------------

SCRIPT_RANGES = [
    ("latin", re.compile(r"[A-Za-zÀ-ɏ]")),
    ("cyrillic", re.compile(r"[Ѐ-ӿԀ-ԯ]")),
    ("armenian", re.compile(r"[԰-֏]")),
    ("georgian", re.compile(r"[Ⴀ-ჿ]")),
    ("arabic", re.compile(r"[؀-ۿ]")),
    ("cjk", re.compile(r"[一-鿿]")),
    ("digit", re.compile(r"[0-9]")),
]


def detect_scripts(s: str):
    """Какие алфавиты встречаются в строке."""
    found = set()
    for name, rx in SCRIPT_RANGES:
        if rx.search(s):
            found.add(name)
    return found


def script_label(s: str) -> str:
    """Один ярлык на строку: если алфавитов несколько - 'mixed'."""
    found = detect_scripts(s) - {"digit"}
    if not found:
        return "other"
    if len(found) == 1:
        return next(iter(found))
    return "mixed:" + "+".join(sorted(found))


# расширенная кириллица: ә, ғ, қ, ң, ө, ұ, ү, һ, і, ў, ҳ, ҷ, ...
EXTRA_CYRILLIC = re.compile(r"[ѐ-ԯӐ-ӿ]")


# ----------------------------------------------------------------------------
# профилирование
# ----------------------------------------------------------------------------


class Report:
    def __init__(self, path):
        self.lines = []
        self.path = path

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def section(self, title):
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def human(n):
    return f"{n:,}".replace(",", " ")


def profile_columns(df, rep, show_values, outdir):
    rep.section("1. СТРУКТУРА")
    rep(f"строк: {human(len(df))}   колонок: {df.shape[1]}")
    rep(f"память: {df.memory_usage(deep=True).sum() / 1024**2:.1f} МБ")
    rep("")
    rep(f"{'колонка':<28} {'тип':<18} {'пропуски':>9} {'уникальных':>12}")
    rep("-" * 72)

    profile = {}
    for c in df.columns:
        s = df[c]
        nulls = s.isna().mean()
        try:
            nuniq = s.nunique(dropna=True)
        except TypeError:  # нехешируемые значения
            nuniq = -1
        profile[c] = {"nulls": nulls, "nunique": nuniq, "dtype": str(s.dtype)}
        rep(f"{c:<28} {str(s.dtype):<18} {nulls:>8.1%} {human(nuniq):>12}")
    return profile


def report_low_cardinality(df, profile, rep, show_values, limit=50):
    rep.section("2. СПРАВОЧНЫЕ КОЛОНКИ (малая мощность)")
    if not show_values:
        rep("значения скрыты флагом --no-values")
        return
    shown = False
    for c, p in profile.items():
        if 0 <= p["nunique"] <= limit and p["nunique"] > 0:
            shown = True
            rep("")
            rep(f"--- {c} ({p['nunique']} значений)")
            vc = df[c].value_counts(dropna=False).head(limit)
            for val, cnt in vc.items():
                rep(f"    {str(val):<30} {human(cnt):>12}  {cnt / len(df):>7.2%}")
    if not shown:
        rep("нет колонок с мощностью <= " + str(limit))


ID_LIKE = re.compile(r"^[A-Za-z0-9_\-.:]+$")


def looks_like_id(s: pd.Series, probe=5000) -> bool:
    """Отличить хеш/идентификатор от имени: без пробелов и только ASCII-алфанум."""
    v = s.dropna().astype(str)
    if len(v) == 0:
        return True
    if len(v) > probe:
        v = v.sample(probe, random_state=0)
    no_space = (~v.str.contains(" ")).mean()
    ascii_only = v.map(lambda x: bool(ID_LIKE.match(x))).mean()
    return no_space > 0.95 and ascii_only > 0.95


def pick_text_columns(df, profile, min_unique_ratio=0.01):
    """Текстовые колонки с высокой вариативностью - кандидаты в имена/названия."""
    cols = []
    for c, p in profile.items():
        if p["dtype"] not in ("object", "string", "str"):
            continue
        if p["nunique"] < 0:
            continue
        if p["nunique"] / max(len(df), 1) < min_unique_ratio:
            continue
        if looks_like_id(df[c]):
            continue
        cols.append(c)
    return cols


def analyse_text(df, col, rep, outdir, sample=200_000):
    s = df[col].dropna().astype(str)
    if len(s) == 0:
        return None
    if len(s) > sample:
        s = s.sample(sample, random_state=0)

    rep("")
    rep(f"--- {col}")
    rep(f"    непустых: {human(int(df[col].notna().sum()))}")

    lengths = s.str.len()
    rep(
        f"    длина: min {lengths.min()}, p50 {int(lengths.median())}, "
        f"p95 {int(lengths.quantile(0.95))}, max {lengths.max()}"
    )
    toks = s.str.split().str.len()
    rep(
        f"    токенов: p50 {int(toks.median())}, p95 {int(toks.quantile(0.95))}, "
        f"max {int(toks.max())}"
    )

    short = (toks <= 1).mean()
    rep(f"    однотокенных (риск ложных совпадений): {short:.2%}")

    labels = s.map(script_label)
    dist = labels.value_counts(normalize=True)
    rep("    алфавиты:")
    for k, v in dist.head(10).items():
        rep(f"        {k:<24} {v:>7.2%}")

    extra = s.str.contains(EXTRA_CYRILLIC).mean()
    rep(f"    нестандартная кириллица (ә, қ, ң, ө, ұ, і, ў, ҳ...): {extra:.2%}")

    dup_exact = 1 - s.nunique() / len(s)
    rep(f"    точных повторов строки: {dup_exact:.2%}")

    # график распределения алфавитов
    fig, ax = plt.subplots(figsize=(7, 4))
    dist.head(8).sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"Алфавиты в поле «{col}»")
    ax.set_xlabel("доля записей")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"scripts_{safe(col)}.png"), dpi=140)
    plt.close(fig)

    # график распределения длин
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths.clip(upper=lengths.quantile(0.99)), bins=60, color="#55A868")
    ax.set_title(f"Длина значения в поле «{col}» (обрезано по p99)")
    ax.set_xlabel("символов")
    ax.set_ylabel("записей")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"length_{safe(col)}.png"), dpi=140)
    plt.close(fig)

    return dist


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def find_gold_id(df, profile, max_filled=0.5):
    """
    Кандидат на роль gold-разметки: колонка-идентификатор, заполненная ЧАСТИЧНО,
    в которой одно значение повторяется у нескольких строк.

    Заполненные почти полностью колонки исключаются намеренно: такой
    идентификатор описывает не «одна сущность = много дублей», а внешний ключ
    (например, id компании, у которой много связанных лиц). Кластеры по нему
    дублями не являются.
    """
    best, best_score = None, 0
    for c, p in profile.items():
        if p["nunique"] <= 1 or p["nunique"] < 0:
            continue
        filled = 1 - p["nulls"]
        if filled >= max_filled or filled <= 0.001:
            continue
        n_filled = int(df[c].notna().sum())
        if n_filled == 0:
            continue
        repeat = 1 - p["nunique"] / n_filled  # доля «лишних» записей в кластерах
        if repeat <= 0.05:
            continue
        score = repeat * min(filled * 20, 1)
        if "id" in c.lower():
            score *= 2
        if score > best_score:
            best, best_score = c, score
    return best


def analyse_clusters(df, col, rep, outdir):
    rep.section(f"4. GOLD-РАЗМЕТКА: «{col}»")
    sub = df[df[col].notna()]
    sizes = sub.groupby(col).size()

    rep(f"размечено записей:      {human(len(sub))}  ({len(sub) / len(df):.2%} базы)")
    rep(f"уникальных сущностей:   {human(int(sizes.shape[0]))}")
    multi = sizes[sizes > 1]
    rep(
        f"сущностей с 2+ записями: {human(int(multi.shape[0]))} "
        f"({multi.shape[0] / max(sizes.shape[0], 1):.1%} от размеченных)"
    )
    rep(f"записей внутри дублей:  {human(int(multi.sum()))}")
    rep("")
    rep("размер кластера:")
    for q in (0.5, 0.9, 0.95, 0.99, 0.999):
        rep(f"    p{q * 100:<6g} {int(sizes.quantile(q))}")
    rep(f"    max     {human(int(sizes.max()))}")

    # сколько пар породит наивная генерация
    pairs = (sizes * (sizes - 1) // 2).sum()
    rep("")
    rep(f"позитивных пар без ограничения: {human(int(pairs))}")
    top = sizes.nlargest(1)
    top_n = int(top.iloc[0])
    rep(
        f"вклад самого крупного кластера ({human(top_n)} записей): "
        f"{human(top_n * (top_n - 1) // 2)} пар "
        f"({top_n * (top_n - 1) // 2 / max(pairs, 1):.1%} от всех)"
    )
    capped = np.minimum(sizes * (sizes - 1) // 2, 100).sum()
    rep(f"с ограничением 100 пар на сущность: {human(int(capped))}")

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = sizes.value_counts().sort_index()
    ax.bar(counts.index[:20], counts.values[:20], color="#C44E52")
    ax.set_yscale("log")
    ax.set_title("Распределение размеров кластеров (gold)")
    ax.set_xlabel("записей в сущности")
    ax.set_ylabel("сущностей (log)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cluster_sizes.png"), dpi=140)
    plt.close(fig)

    return sub, sizes


def analyse_cluster_difficulty(sub, gold, rep, outdir, name_col=None, country_col=None):
    """
    Доля дублей, решаемых точным совпадением нормализованной строки.
    """
    rep.section("4а. СЛОЖНОСТЬ ДУБЛЕЙ")
    multi = sub.groupby(gold).filter(lambda g: len(g) > 1)
    if len(multi) == 0:
        rep("кластеров с 2+ записями нет")
        return

    rep(f"записей в кластерах 2+: {human(len(multi))}")

    if name_col and name_col in multi.columns:
        norm = (
            multi[name_col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.split()
            .str.join(" ")
        )
        tmp = pd.DataFrame({"g": multi[gold].values, "n": norm.values})
        per = tmp.groupby("g")["n"].nunique()
        size = tmp.groupby("g").size()
        identical = (per == 1).mean()
        rep("")
        rep(f"поле сравнения: «{name_col}» (регистр и пунктуация убраны)")
        rep(f"    кластеров, где все написания идентичны: {identical:.1%}")
        rep(f"    кластеров с 2+ вариантами написания:    {1 - identical:.1%}")
        rep(
            f"    в среднем вариантов написания на кластер: "
            f"{(per / size).mean():.2f} от размера"
        )
        rep("")

    if country_col and country_col in multi.columns:
        per_c = multi.groupby(gold)[country_col].nunique()
        rep("")
        rep(f"    кластеров, охватывающих 2+ страны: {(per_c > 1).mean():.2%}")
        rep("    (cross-country кейсы)")

    if name_col and name_col in multi.columns:
        scripts = multi[name_col].astype(str).map(script_label)
        tmp = pd.DataFrame({"g": multi[gold].values, "s": scripts.values})
        per_s = tmp.groupby("g")["s"].nunique()
        rep(f"    кластеров с записями в 2+ алфавитах: {(per_s > 1).mean():.2%}")


def cross_tab(df, profile, rep, outdir, max_card=50):
    """Двумерные разрезы по справочным колонкам - ищем перекосы."""
    cats = [c for c, p in profile.items() if 1 < p["nunique"] <= max_card]
    if len(cats) < 1:
        return
    rep.section("5. ПЕРЕКОСЫ ПО КАТЕГОРИЯМ")
    for c in cats[:4]:
        vc = df[c].value_counts(normalize=True, dropna=False)
        top3 = vc.head(3).sum()
        rep("")
        rep(f"--- {c}: три крупнейшие категории дают {top3:.1%} базы")
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.head(15).sort_values().plot.barh(ax=ax, color="#8172B2")
        ax.set_title(f"Распределение по «{c}»")
        ax.set_xlabel("доля записей")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"dist_{safe(c)}.png"), dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="путь к parquet/csv с датасетом")
    ap.add_argument("--out", default="eda_out")
    ap.add_argument(
        "--no-values",
        action="store_true",
        help="не печатать значения даже справочных колонок",
    )
    ap.add_argument("--rows", type=int, default=0, help="читать только N строк")
    ap.add_argument(
        "--gold",
        default=None,
        help="колонка gold-разметки (если не указана, определяется автоматически)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rep = Report(os.path.join(args.out, "report.txt"))

    rep(f"файл: {os.path.basename(args.data)}")
    rep(f"размер на диске: {os.path.getsize(args.data) / 1024**2:.1f} МБ")

    if args.data.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    if args.rows:
        df = df.head(args.rows)

    profile = profile_columns(df, rep, not args.no_values, args.out)
    report_low_cardinality(df, profile, rep, not args.no_values)

    rep.section("3. ТЕКСТОВЫЕ ПОЛЯ (имена и названия)")
    text_cols = pick_text_columns(df, profile)
    if not text_cols:
        rep("не найдено текстовых колонок с высокой вариативностью")
    for c in text_cols[:6]:
        analyse_text(df, c, rep, args.out)

    gold = args.gold or find_gold_id(df, profile)
    if gold and gold not in df.columns:
        rep(f"колонка «{gold}» не найдена, беру автоопределение")
        gold = find_gold_id(df, profile)
    if gold:
        sub, _ = analyse_clusters(df, gold, rep, args.out)
        name_col = text_cols[0] if text_cols else None
        for cand in ("party_name", "name", "full_name"):
            if cand in df.columns:
                name_col = cand
                break
        country_col = next(
            (c for c in df.columns if c.lower() in ("country", "country_code", "iso")),
            None,
        )
        analyse_cluster_difficulty(sub, gold, rep, args.out, name_col, country_col)
    else:
        rep.section("4. GOLD-РАЗМЕТКА")
        rep("частично заполненной колонки-идентификатора не найдено -")
        rep("посмотрите таблицу в разделе 1 и укажите её вручную")

    cross_tab(df, profile, rep, args.out)

    rep.section("ИТОГ")
    rep(f"графики: {os.path.relpath(args.out)}")
    rep(f"отчёт:   {os.path.relpath(rep.path)}")
    rep("")
    rep.save()


if __name__ == "__main__":
    main()
