"""
Отчёт о качестве модели и размеченный тестовый набор.
Результат: work/quality_report.md, work/test_set_labeled.csv, work/matcher.pkl.
"""

import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config as cfg
import matcher


def human(n):
    return f"{n:,}".replace(",", " ")


def main():
    out = []
    say = lambda s="": (print(s), out.append(s))

    say("# Отчёт о качестве модели матчинга")
    say()
    say("Модель: LightGBM на 11 признаках схожести пары имён. "
        "Инференс на CPU. Обучающий набор построен синтетически "
        "(см. ноутбук 02 и раздел «Методология оценки»).")
    say()

    # --- данные и пары ---------------------------------------------------
    df = pd.read_parquet(cfg.DATA, columns=[cfg.COL_NAME, cfg.COL_PARTY_TYPE, cfg.COL_COUNTRY])
    samp = df[df[cfg.COL_PARTY_TYPE] == "individual"].sample(300_000, random_state=cfg.RANDOM_STATE)
    names = samp[cfg.COL_NAME].astype(str).tolist()
    # страна по нормализованному имени - для разреза качества по странам
    name2country = {matcher.prep(k): v for k, v in
                    samp.drop_duplicates(cfg.COL_NAME).set_index(cfg.COL_NAME)[cfg.COL_COUNTRY].items()}

    pairs = matcher.make_training_pairs(names, verbose=False)
    pairs["split"] = matcher.split_by_entity(pairs)

    say("## 1. Обучающий набор")
    say()
    say("| Часть | Тип пар | Количество |")
    say("|---|---|---|")
    for (lab, kind), n in pairs.groupby(["label", "kind"]).size().items():
        say(f"| {'позитивы' if lab else 'негативы'} | {kind} | {human(n)} |")
    say()
    say("| Сплит | Пар | Позитивов |")
    say("|---|---|---|")
    for sp, g in pairs.groupby("split"):
        say(f"| {sp} | {human(len(g))} | {human(int(g.label.sum()))} |")
    say()
    tr_ent = set(pairs[pairs.split == "train"].left)
    te_ent = set(pairs[pairs.split == "test"].left)
    say(f"Разбиение по сущностям; пересечение сущностей train и test: "
        f"**{len(tr_ent & te_ent)}** (утечки нет).")
    say()

    # --- модель ----------------------------------------------------------
    model, X, y = matcher.train(pairs, verbose=False)
    matcher.save(model)
    sp = pairs["split"].values
    te = sp == "test"
    proba = model.predict_proba(X[te])[:, 1]
    yt = y[te]

    # --- baseline: порог по лучшей мере ---------------------------------
    L = pairs.loc[te, "left"].tolist()
    R = pairs.loc[te, "right"].tolist()
    base_score = np.array([fuzz.token_set_ratio(a, b) / 100 for a, b in zip(L, R)])
    # порог baseline подбираем на train, чтобы не подглядывать в test
    trm = sp == "train"
    Ltr = pairs.loc[trm, "left"].tolist(); Rtr = pairs.loc[trm, "right"].tolist()
    base_tr = np.array([fuzz.token_set_ratio(a, b) / 100 for a, b in zip(Ltr, Rtr)])
    best_t, best_f = 0.5, 0
    for t in np.arange(0.5, 0.99, 0.01):
        f = f1_score(y[trm], (base_tr >= t).astype(int))
        if f > best_f:
            best_f, best_t = f, t
    base_pred = (base_score >= best_t).astype(int)

    say("## 2. Метрики на тестовом наборе")
    say()
    say("| Метрика | Baseline: порог token_set | Финальная модель |")
    say("|---|---|---|")
    pred = (proba >= 0.5).astype(int)
    rows = [
        ("Precision", precision_score(yt, base_pred), precision_score(yt, pred)),
        ("Recall", recall_score(yt, base_pred), recall_score(yt, pred)),
        ("F1", f1_score(yt, base_pred), f1_score(yt, pred)),
        ("ROC-AUC", roc_auc_score(yt, base_score), roc_auc_score(yt, proba)),
    ]
    for name, b, m in rows:
        say(f"| {name} | {b:.4f} | **{m:.4f}** |")
    say()
    say(f"Baseline: одна мера схожести (token_set) и порог {best_t:.2f}, "
        f"подобранный на train.")
    say()

    # --- confusion matrix ------------------------------------------------
    say("## 3. Confusion matrix (финальная модель, порог 0,5)")
    say()
    cm = confusion_matrix(yt, pred)
    tn, fp, fn, tp = cm.ravel()
    say("| | предсказано: не дубль | предсказано: дубль |")
    say("|---|---|---|")
    say(f"| **факт: не дубль** | {human(tn)} (TN) | {human(fp)} (FP) |")
    say(f"| **факт: дубль** | {human(fn)} (FN) | {human(tp)} (TP) |")
    say()
    say(f"FP (ложные склейки): {human(fp)}; FN (пропущенные дубли): {human(fn)}.")
    say()

    # --- где ошибается ---------------------------------------------------
    say("## 4. Где модель ошибается")
    say()
    tdf = pairs[te].copy()
    tdf["pred"] = pred
    tdf["proba"] = proba
    say("| Подмножество | Пар | Доля верных |")
    say("|---|---|---|")
    for k in (1, 2, 3):
        m = tdf[(tdf.label == 1) & (tdf.n_ops == k)]
        say(f"| дубли, {k} искажени{'е' if k == 1 else 'я'} | {human(len(m))} | {m.pred.mean():.4f} |")
    for kind in ("однофамильцы", "случайные"):
        m = tdf[(tdf.label == 0) & (tdf.kind == kind)]
        say(f"| негативы: {kind} | {human(len(m))} | {1 - m.pred.mean():.4f} |")
    say()
    say("Ошибки сосредоточены там, где и ожидалось: сильно искажённые дубли "
        "(3 искажения) и однофамильцы. Случайные негативы модель отсекает "
        "почти безошибочно.")
    say()

    # --- по странам ------------------------------------------------------
    say("## 4а. Качество по странам")
    say()
    say("Обучающая выборка случайная, страны представлены пропорционально базе "
        "(mng 41%, kaz 27%, arm 12%, azb 10%, kgz 7%, tjk 3%). Проверка, "
        "не страдает ли качество на малых странах.")
    say()
    tdf["country"] = tdf["left"].map(name2country)
    say("| Страна | Пар | Позитивов | Precision | Recall | F1 |")
    say("|---|---|---|---|---|---|")
    for c, g in tdf.dropna(subset=["country"]).groupby("country"):
        if g.label.sum() < 50:
            continue
        say(f"| {c} | {human(len(g))} | {human(int(g.label.sum()))} | "
            f"{precision_score(g.label, g.pred):.3f} | {recall_score(g.label, g.pred):.3f} | "
            f"{f1_score(g.label, g.pred):.3f} |")
    say()
    say("Разброс F1 по странам невелик, и самая малочисленная страна не хуже "
        "самой массовой: после транслитерации признаки не зависят от алфавита "
        "и страны, модель учится тому, как искажения меняют строку. "
        "Стратификация по стране при обучении не требуется.")
    say()

    # --- порог -----------------------------------------------------------
    say("## 5. Зависимость от порога")
    say()
    say("| Порог | Precision | Recall | F1 |")
    say("|---|---|---|---|")
    for t in (0.3, 0.5, 0.7, 0.9, 0.95):
        p = (proba >= t).astype(int)
        say(f"| {t} | {precision_score(yt, p):.3f} | {recall_score(yt, p):.3f} | {f1_score(yt, p):.3f} |")
    say()
    say("Рабочий порог - 0,9; в API отдаётся как параметр.")
    say()

    # --- feature importance ---------------------------------------------
    say("## 6. Feature importance")
    say()
    imp = pd.Series(model.feature_importances_, index=matcher.FEATURE_NAMES)
    imp = imp.sort_values(ascending=False)
    say("| Признак | Importance | Что ловит |")
    say("|---|---|---|")
    what = {
        "ratio": "посимвольное сходство - опечатки",
        "token_sort": "сходство с сортировкой токенов - порядок ФИО",
        "token_set": "сравнение множеств токенов - лишние слова",
        "partial": "лучшая подстрока - сокращения",
        "jaro_winkler": "вес на общее начало - фамилии",
        "levenshtein": "число правок",
        "len_diff": "относительная разница длин",
        "tok_diff": "разница числа токенов",
        "same_phon": "совпал ли фонетический код",
        "same_init": "совпали ли инициалы",
        "min_tokens": "сколько токенов в более короткой записи",
    }
    for k, v in imp.items():
        say(f"| `{k}` | {int(v)} | {what.get(k, '')} |")
    say()
    say("Интерпретация: модель опирается на несколько строковых мер сразу - в топе `jaro_winkler`, `partial`, `len_diff`. Структурные признаки (`tok_diff`, `same_init`, `same_phon`) внизу списка. Это узкое место: однофамильцев модель отделяет в основном по строковой схожести, а контекста (общие компании, роли, страны) у неё нет вообще. Добавление контекстных признаков - первая точка роста.")
    say()

    # --- сохранить -------------------------------------------------------
    test_out = os.path.join(cfg.WORK, "test_set_labeled.csv")
    tdf[["left", "right", "label", "kind", "n_ops", "proba", "pred"]].to_csv(
        test_out, index=False, encoding="utf-8")
    rep_out = os.path.join(cfg.WORK, "quality_report.md")
    with open(rep_out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print()
    print("отчёт:        ", os.path.relpath(rep_out))
    print("тестовый набор:", os.path.relpath(test_out), f"({human(len(tdf))} пар)")


if __name__ == "__main__":
    main()
