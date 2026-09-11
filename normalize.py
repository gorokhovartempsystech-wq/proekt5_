"""Нормализация имён, фонетическая транслитерация в латиницу, извлечение юридических форм."""

import re
import unicodedata

import jellyfish

# ---------------------------------------------------------------------------
# таблицы транслитерации
# ---------------------------------------------------------------------------

# базовая кириллица
CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# расширенная кириллица: казахский, киргизский, монгольский, таджикский, узбекский.
CYR_EXTRA = {
    "ә": "a", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
    "һ": "h", "і": "i", "ї": "i", "ў": "u", "ҳ": "h", "ҷ": "j", "ҹ": "j",
    "ҫ": "s", "ҙ": "z", "ҡ": "k", "ҥ": "n", "ӑ": "a", "ӓ": "a",
    "ӗ": "e", "ӝ": "zh", "ӟ": "z", "ӥ": "i", "ӧ": "o", "ӱ": "u", "ӳ": "u",
    "ӹ": "y", "ѓ": "g", "ѕ": "z", "ј": "y", "љ": "l", "њ": "n", "ћ": "ch",
    "ќ": "k", "џ": "j", "ҽ": "ch", "ӕ": "ae",
}

# армянский алфавит, фонетически (восточноармянское произношение)
ARM = {
    "ա": "a", "բ": "b", "գ": "g", "դ": "d", "ե": "e", "զ": "z", "է": "e",
    "ը": "y", "թ": "t", "ժ": "zh", "ի": "i", "լ": "l", "խ": "kh", "ծ": "ts",
    "կ": "k", "հ": "h", "ձ": "dz", "ղ": "gh", "ճ": "ch", "մ": "m", "յ": "y",
    "ն": "n", "շ": "sh", "ո": "o", "չ": "ch", "պ": "p", "ջ": "j", "ռ": "r",
    "ս": "s", "վ": "v", "տ": "t", "ր": "r", "ց": "ts", "ւ": "v", "փ": "p",
    "ք": "k", "օ": "o", "ֆ": "f", "և": "ev",
}

# азербайджанская и турецкая латиница с диакритикой
LAT_DIA = {
    "ə": "a", "ı": "i", "ğ": "g", "ş": "sh", "ç": "ch", "ö": "o", "ü": "u",
    "İ": "i", "â": "a", "ê": "e", "î": "i", "ô": "o", "û": "u", "ñ": "n",
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ý": "y", "ä": "a",
}

TRANSLIT = {}
for table in (CYR, CYR_EXTRA, ARM, LAT_DIA):
    TRANSLIT.update(table)

# ---------------------------------------------------------------------------
# юридические формы
# ---------------------------------------------------------------------------

# Словарь юрформ собран вручную по странам выборки: mng, kgz, kaz, arm, azb, tjk.
LEGAL_FORMS = {
    "llc": ["ооо", "тоо", "жшс", "чп", "пк",
            "llc", "ltd", "limited", "ooo", "too", "co", "company",
            "хх", "ххк", "mchj", "mchc", "мчж",
            "սպը", "спэ",
            "mmc", "ммс"],
    "jsc": ["ао", "оао", "зао", "акционерное", "ак", "jsc", "sa", "aş", "as",
            "asc", "qsc", "бнхаа", "хк", "բաց", "փբը", "пбэ"],
    "ie": ["индивидуальный", "предприниматель", "жеке", "ie", "ип"],
    "gov": ["министерство", "агентство", "департамент", "ministry",
            "agency", "department", "яам"],
    "ngo": ["фонд", "ассоциация", "союз", "foundation", "association",
            "union", "нөхөрлөл", "ohf"],
}
FORM_LOOKUP = {}
for form, words in LEGAL_FORMS.items():
    for w in words:
        FORM_LOOKUP[w] = form

# ---------------------------------------------------------------------------
# определение алфавита
# ---------------------------------------------------------------------------

SCRIPT_RANGES = [
    ("latin", re.compile(r"[A-Za-zÀ-ɏ]")),
    ("cyrillic", re.compile(r"[Ѐ-ӿԀ-ԯ]")),
    ("armenian", re.compile(r"[԰-֏]")),
    ("georgian", re.compile(r"[Ⴀ-ჿ]")),
    ("arabic", re.compile(r"[؀-ۿ]")),
]


def script_label(s: str) -> str:
    """Какой алфавит у строки. Несколько сразу - 'mixed:a+b'."""
    found = {name for name, rx in SCRIPT_RANGES if rx.search(s)}
    if not found:
        return "other"
    if len(found) == 1:
        return next(iter(found))
    return "mixed:" + "+".join(sorted(found))


def script_label_for_series(series):
    """То же для pandas.Series - считаем по уникальным значениям."""
    s = series.astype(str)
    cache = {v: script_label(v) for v in s.unique()}
    return s.map(cache)


PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
SPACES = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")


def normalize(s) -> str:
    """Регистр, юникод, пунктуация, лишние пробелы."""
    if s is None:
        return ""
    s = str(s)
    # NFKC разворачивает лигатуры и совместимые формы, но diacritics
    # сохраняем - их снимет транслитерация, у неё свои правила по языкам
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("ё", "е").replace("«", " ").replace("»", " ")
    s = PUNCT.sub(" ", s)
    s = SPACES.sub(" ", s).strip()
    return s


def translit(s: str) -> str:
    """Фонетическая транслитерация в латиницу."""
    out = []
    for ch in s:
        if ch in TRANSLIT:
            out.append(TRANSLIT[ch])
        elif ch.isascii():
            out.append(ch)
        else:
            # неизвестный символ: пробуем разложить и снять диакритику
            d = unicodedata.normalize("NFD", ch)
            base = "".join(c for c in d if not unicodedata.combining(c))
            out.append(base if base.isascii() else " ")
    return SPACES.sub(" ", "".join(out)).strip()


def sorted_tokens(s: str) -> str:
    """Порядок элементов имени различается по странам - снимаем его."""
    return " ".join(sorted(s.split()))


def phonetic_key(s: str) -> str:
    """Фонетический код имени: metaphone по каждому токену."""
    keys = []
    for t in s.split():
        if not t or not t.isascii():
            continue
        try:
            k = jellyfish.metaphone(t)
        except Exception:
            k = ""
        keys.append(k or t[:4])
    return " ".join(sorted(keys))


def extract_legal_form(s: str):
    """Достаём организационно-правовую форму и убираем её из названия."""
    toks = s.split()
    found, rest = None, []
    for t in toks:
        f = FORM_LOOKUP.get(t)
        if f and found is None:
            found = f
        elif f:
            pass  # повторная форма - тоже выбрасываем
        else:
            rest.append(t)
    return found, " ".join(rest)


def strip_digits(s: str) -> str:
    return SPACES.sub(" ", DIGITS.sub(" ", s)).strip()


def build_keys(raw, is_company=False):
    """Полный набор представлений одной строки."""
    norm = normalize(raw)
    form = None
    if is_company:
        form, norm_nf = extract_legal_form(norm)
    else:
        norm_nf = norm
    lat = translit(norm_nf)
    return {
        "norm": norm,
        "legal_form": form,
        "lat": lat,
        "lat_sorted": sorted_tokens(lat),
        "phon": phonetic_key(lat),
        "initials": "".join(t[0] for t in lat.split() if t),
        "n_tokens": len(lat.split()),
    }


if __name__ == "__main__":
    samples = [
        "Иванов Пётр Сергеевич",
        "IVANOV PETR",
        'ООО "Ромашка"',
        "Ромашка ЖШС",
        "Ғабдуллин Қанат Нұрланұлы",
        "Աբրահամյան Արամ",
        "Əliyev Rəşad",
    ]
    for s in samples:
        k = build_keys(s, is_company="ооо" in s.lower() or "жшс" in s.lower())
        print(f"{s!r:40} -> lat={k['lat']!r:32} sorted={k['lat_sorted']!r:32} "
              f"phon={k['phon']!r} form={k['legal_form']}")
