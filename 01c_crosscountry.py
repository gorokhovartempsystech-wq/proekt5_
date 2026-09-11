"""
Проверка cross-country дублей: покрытие разметкой и точное совпадение имён между странами.
Контрольные числа из summary.json: person_overlap_targets = 3693, company_overlap_targets = 182.
Результат: work/crosscountry_report.txt.
"""

import os
import sys

import pandas as pd

import config as cfg
from normalize import normalize, translit

EXPECTED = {"person": 3693, "company": 182}


def human(n):
    return f"{n:,}".replace(",", " ")


def build_key(series: pd.Series) -> pd.Series:
    """Нормализация + транслитерация, посчитанные по уникальным значениям."""
    s = series.dropna().astype(str)
    cache = {v: translit(normalize(v)) for v in s.unique()}
    return s.map(cache)


def scan(df, values, label, expected, rep):
    """Сколько ключей встречается более чем в одной стране."""
    key = build_key(values)
    tab = pd.DataFrame({"k": key.values, "c": df.loc[key.index, cfg.COL_COUNTRY].values})
    tab = tab[tab.k.str.len() > 0]

    per = tab.groupby("k")["c"].nunique()
    multi = per[per > 1]

    rep("")
    rep(f"--- {label}")
    rep(f"    уникальных нормализованных значений: {human(len(per))}")
    rep(f"    встречаются в 2+ странах:            {human(len(multi))}")
    if expected:
        rep(f"    заявлено в summary.json:             {human(expected)}")
        rep(f"    разница:                             {human(abs(len(multi) - expected))}")
    if len(multi):
        rep("    распределение по числу стран:")
        for k, v in multi.value_counts().sort_index().items():
            rep(f"        {k} страны: {human(v)}")
    return multi


def main():
    lines = []

    def rep(*p):
        s = " ".join(str(x) for x in p)
        print(s, flush=True)
        lines.append(s)

    rep("=" * 74)
    rep("CROSS-COUNTRY ДУБЛИ")
    rep("=" * 74)

    df = pd.read_parquet(
        cfg.DATA,
        columns=[cfg.COL_COUNTRY, cfg.COL_PARTY_TYPE, cfg.COL_NAME,
                 cfg.COL_COMPANY_NAME, cfg.COL_GOLD],
    )
    rep(f"строк: {human(len(df))}")

    # 1. подтверждаем, что gold не покрывает cross-country
    g = df[df[cfg.COL_GOLD].notna()]
    spans = g.groupby(cfg.COL_GOLD)[cfg.COL_COUNTRY].nunique()
    rep("")
    rep("1. ПОКРЫТИЕ GOLD-РАЗМЕТКИ")
    rep(f"    идентификаторов в разметке:       {human(len(spans))}")
    rep(f"    из них охватывают 2+ страны:      {human(int((spans > 1).sum()))}")
    rep("    (surrogate key построен внутри страны)")

    # 2. сколько cross-country совпадений даёт тривиальный baseline
    rep("")
    rep("2. BASELINE: точное совпадение после транслитерации")

    ind = df[df[cfg.COL_PARTY_TYPE] == "individual"]
    m_person = scan(ind, ind[cfg.COL_NAME], "физлица (party_name)", EXPECTED["person"], rep)

    le = df[df[cfg.COL_PARTY_TYPE] == "legal_entity"]
    scan(le, le[cfg.COL_NAME], "юрлица на стороне party", None, rep)

    comp = df[df[cfg.COL_COMPANY_NAME].notna()]
    m_comp = scan(comp, comp[cfg.COL_COMPANY_NAME], "компании (company_name_norm)",
                  EXPECTED["company"], rep)

    # 3. выводы
    rep("")
    rep("=" * 74)
    rep("ЧТО ИЗ ЭТОГО СЛЕДУЕТ")
    rep("=" * 74)
    rep("")
    rep(f"Физлица: точное совпадение находит {human(len(m_person))} сущностей "
        f"против {human(EXPECTED['person'])} заложенных.")
    rep("")
    rep(f"Компании: точное совпадение находит {human(len(m_comp))} против "
        f"{human(EXPECTED['company'])} заложенных, то есть в разы больше.")
    rep("    (превышение: типовые названия совпадают у разных юрлиц)")
    rep("")
    rep("Отсюда методология оценки строится на трёх наборах:")
    rep("    A. gold          - внутристрановая разметка; проверка, что пайплайн")
    rep("                       не разрывает известные сущности;")
    rep("    B. синтетика     - искажения реальных имён с известной меткой;")
    rep("                       основные метрики precision / recall / F1;")
    rep("    C. cross-country - меток нет, есть контрольные числа 3693 и 182;")
    rep("                       по ним проверяется порог.")
    rep("")

    out = os.path.join(cfg.WORK, "crosscountry_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    rep("")
    rep(f"отчёт: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
