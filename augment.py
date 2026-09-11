"""Синтетические искажения имён для генерации позитивных пар."""

import random
import re

from normalize import CYR, CYR_EXTRA, normalize, translit

# ---------------------------------------------------------------------------
# 1. Транслитерация через другой стандарт
# ---------------------------------------------------------------------------

# Одна и та же буква в разных источниках латинизируется по-разному.
# Слева - как у нас, справа - альтернативы, встречающиеся в реальных выгрузках.
ALT_TRANSLIT = {
    "zh": ["j", "g", "z"],
    "kh": ["h", "x", "k"],
    "ts": ["c", "tz"],
    "ch": ["c", "tch"],
    "sh": ["s", "sch"],
    "yu": ["iu", "ju", "u"],
    "ya": ["ia", "ja", "a"],
    "y": ["i", "j"],
    "i": ["y"],
    "ov": ["ow", "off", "of"],
    "ev": ["ew", "eff", "ef"],
    "ks": ["x"],
    "g": ["gh"],
    "k": ["q", "c"],
    "u": ["oo", "ou"],
    "e": ["ye"],
}


def alt_translit(s: str, rng: random.Random, n: int = 1) -> str:
    """Заменяет один-два сочетания на альтернативную латинизацию."""
    keys = [k for k in ALT_TRANSLIT if k in s]
    if not keys:
        return s
    rng.shuffle(keys)
    for k in keys[:n]:
        s = s.replace(k, rng.choice(ALT_TRANSLIT[k]), 1)
    return s


# ---------------------------------------------------------------------------
# 2. Потеря диакритики и расширенной кириллицы
# ---------------------------------------------------------------------------

# Часть источников выгружает текст в ASCII, теряя специфические буквы.
FOLD = {"ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у",
        "ү": "у", "һ": "х", "і": "и", "ў": "у", "ҳ": "х", "ҷ": "ч",
        "ə": "a", "ı": "i", "ğ": "g", "ş": "s", "ç": "c", "ö": "o", "ü": "u"}


def fold_diacritics(s: str, rng: random.Random) -> str:
    return "".join(FOLD.get(ch, ch) for ch in s)


# ---------------------------------------------------------------------------
# 3. Порядок и состав токенов
# ---------------------------------------------------------------------------


def swap_tokens(s: str, rng: random.Random) -> str:
    """«Фамилия Имя» - «Имя Фамилия» - порядок различается по странам."""
    t = s.split()
    if len(t) < 2:
        return s
    i = rng.randrange(len(t) - 1)
    t[i], t[i + 1] = t[i + 1], t[i]
    return " ".join(t)


def drop_token(s: str, rng: random.Random) -> str:
    """Отчество или второе имя часто отсутствует в одном из источников."""
    t = s.split()
    if len(t) < 3:
        return s
    del t[rng.randrange(1, len(t))]
    return " ".join(t)


def abbreviate(s: str, rng: random.Random) -> str:
    """«Иванов Пётр Сергеевич» -> «Иванов П С»."""
    t = s.split()
    if len(t) < 2:
        return s
    return " ".join([t[0]] + [x[0] for x in t[1:]])


# ---------------------------------------------------------------------------
# 4. Опечатки
# ---------------------------------------------------------------------------

KEYBOARD = {
    "а": "фвыпр", "б": "юьди", "в": "чсмуа", "г": "нршо", "д": "жлбщ",
    "е": "нкупр", "и": "мтьбю", "к": "епаув", "л": "дощж", "м": "сивчя",
    "н": "гершо", "о": "лрщтг", "п": "аривк", "р": "поенг", "с": "чвмыа",
    "т": "иьбющ", "у": "цвкае", "ы": "фсчвз", "a": "qwsz", "e": "wrsd",
    "i": "uojk", "o": "ipkl", "n": "bhjm", "r": "etdf", "s": "adwx",
    "t": "ryfg", "l": "kop", "k": "jlio", "v": "cbfg", "m": "n,jk",
}


def typo(s: str, rng: random.Random) -> str:
    """Одна опечатка: соседняя клавиша, пропуск, удвоение или перестановка."""
    if len(s) < 4:
        return s
    kind = rng.choice(["sub", "del", "dup", "swap"])
    i = rng.randrange(len(s))
    ch = s[i].lower()
    if kind == "sub" and ch in KEYBOARD:
        return s[:i] + rng.choice(KEYBOARD[ch]) + s[i + 1:]
    if kind == "del":
        return s[:i] + s[i + 1:]
    if kind == "dup":
        return s[:i] + s[i] + s[i:]
    if kind == "swap" and i < len(s) - 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


# ---------------------------------------------------------------------------
# 5. Юридические формы (для компаний)
# ---------------------------------------------------------------------------

FORM_VARIANTS = [
    ["ооо", "ооо", "общество с ограниченной ответственностью", "llc", "ltd"],
    ["тоо", "жшс", "too", "llc"],
    ["ххк", "хк", "llc"],
    ["ао", "акционерное общество", "jsc"],
]


def swap_legal_form(s: str, rng: random.Random) -> str:
    """Заменяет юрформу на эквивалентную из другой страны или языка."""
    low = s.lower()
    for group in FORM_VARIANTS:
        for form in group:
            if re.search(rf"\b{re.escape(form)}\b", low):
                new = rng.choice([g for g in group if g != form])
                return re.sub(rf"\b{re.escape(form)}\b", new, low, count=1)
    return s


# ---------------------------------------------------------------------------
# набор искажений
# ---------------------------------------------------------------------------

def to_latin_variant(s: str, rng: random.Random) -> str:
    """Один источник хранит имя в родном алфавите, другой - в латинице по своему стандарту."""
    lat = translit(normalize(s))
    return alt_translit(lat, rng, n=rng.choice([1, 2]))


CORRUPTIONS = {
    "to_latin_variant": to_latin_variant,   # смена алфавита
    "alt_translit": alt_translit,           # другая латинизация (уже на латинице)
    "fold_diacritics": fold_diacritics,     # потеря спецбукв
    "swap_tokens": swap_tokens,             # другой порядок ФИО
    "drop_token": drop_token,               # нет отчества
    "abbreviate": abbreviate,               # инициалы
    "typo": typo,                           # опечатка
    "swap_legal_form": swap_legal_form,     # другая юрформа
}

# Веса подобраны так, чтобы частые в жизни искажения встречались чаще.
DEFAULT_WEIGHTS = {
    "to_latin_variant": 3,
    "alt_translit": 2,
    "fold_diacritics": 3,
    "swap_tokens": 2,
    "drop_token": 2,
    "abbreviate": 1,
    "typo": 2,
    "swap_legal_form": 1,
}

MAX_TRIES = 12


def corrupt(name: str, n_ops: int = 1, seed=None, weights=None, allowed=None):
    """
    Возвращает (искажённое_имя, список_применённых_искажений).

    n_ops - сколько искажений применить подряд. Одно - лёгкий кейс,
    три - тяжёлый. Этим параметром задаётся управляемая сложность,
    и по нему потом строится кривая «качество против степени искажения».

    Если выбранное искажение к строке неприменимо (например, смена юрформы
    у имени человека), берём следующее - иначе на выходе получались бы
    неискажённые пары, и сложность набора была бы не той, что заявлена.
    """
    rng = random.Random(seed)
    weights = weights or DEFAULT_WEIGHTS
    pool = [k for k in CORRUPTIONS if (allowed is None or k in allowed)]
    if not pool:
        return name, []
    w = [weights.get(k, 1) for k in pool]

    out, applied = name, []
    for _ in range(n_ops):
        for _try in range(MAX_TRIES):
            op = rng.choices(pool, weights=w, k=1)[0]
            new = CORRUPTIONS[op](out, rng)
            if new and new != out:
                applied.append(op)
                out = new
                break
    return out, applied


if __name__ == "__main__":
    demo = [
        "Иванов Пётр Сергеевич",
        "Ғабдуллин Қанат Нұрланұлы",
        'ООО "Ромашка"',
    ]
    for name in demo:
        print(f"\n{name}")
        for k in range(1, 4):
            out, ops = corrupt(name, n_ops=k, seed=k)
            print(f"  {k} искажени(я): {out!r:45} {ops}")
