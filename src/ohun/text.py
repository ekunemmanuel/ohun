"""Text normalization for training and scoring.

Yoruba is written with under-dots (ẹ ọ ṣ) and tone marks (high ´, low `, mid unmarked).
Real-world text is inconsistent: some sources drop the marks, some use a vertical line
below instead of a dot, and some write characters precomposed while others don't. If two
spellings of the same word differ only in how they were typed, the model and the scorer
must see them as identical. If they differ in tone marks, they are different words
(ọkọ́ = hoe, ọkọ̀ = vehicle, ọkọ = husband) and that difference must survive.

Levels for Yoruba:
  "full"    keep under-dots and tones (what we train on and the headline score)
  "no_tone" keep under-dots, drop tones (how much error is only tone)
  "bare"    plain ASCII-ish letters (comparable to sources with no marks at all)
"""

import re
import unicodedata
from typing import Literal

Level = Literal["full", "no_tone", "bare"]

ACUTE = "́"  # high tone
GRAVE = "̀"  # low tone
MACRON = "̄"  # explicit mid tone, rare
DOT_BELOW = "̣"  # ẹ ọ ṣ
VERTICAL_LINE_BELOW = "̩"  # e̩ o̩ s̩, an older way of writing the under-dot
TONES = {ACUTE, GRAVE, MACRON}

# Punctuation we drop for scoring. Apostrophes and hyphens inside a word are kept.
# Combining marks (U+0300-U+036F) are not \w in Python, so they are listed explicitly:
# ọ̀ has no precomposed form and its low tone would otherwise be stripped as punctuation.
_PUNCT = re.compile(r"[^\w\s'\-̀-ͯ]|(?<!\w)['\-]|['\-](?!\w)")
_SPACES = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "–": "-", "—": " "})

# Nigerian Pidgin has no fixed spelling. BBC Pidgin (which most datasets follow) writes
# "pipo", "di", "moni"; English-trained models write "people", "the", "money". Same words,
# so the scorer maps both to one form. Only add a pair when it is unambiguously the same
# word: "dem" stays out because it is both "them" and the plural marker.
# Grow this from real transcripts, one pair at a time.
PIDGIN_VARIANTS = {
    # spelling variants within Pidgin
    "wettin": "wetin",
    "pickin": "pikin",
    "abeeg": "abeg",
    # BBC Pidgin spelling -> English spelling of the same word
    "di": "the",
    "dis": "this",
    "dat": "that",
    "pipo": "people",
    "plenti": "plenty",
    "wit": "with",
    "dia": "their",
    "moni": "money",
    "tori": "story",
    "epp": "help",
    "afta": "after",
    "bifor": "before",
    "oda": "other",
    "broda": "brother",
    "sista": "sister",
    "pesin": "person",
    "kontri": "country",
    "govment": "government",
    "polis": "police",
    "notin": "nothing",
    "somtin": "something",
    "evritin": "everything",
    "nobodi": "nobody",
    "evribodi": "everybody",
    "tin": "thing",
    "tins": "things",
    "wia": "where",
    "hapun": "happen",
    "tok": "talk",
    "wen": "when",
    "becos": "because",
    "bikos": "because",
    "govnor": "governor",
    "hin": "him",
    # Naija phonetic spelling (what MMS writes) -> the common spelling
    "sey": "say",
    "pipol": "people",
    "yu": "you",
    "wi": "we",
    "nor": "no",
    "de": "dey",
    "komot": "comot",
    "koll": "call",
    "kom": "come",
    "com": "come",
}


def normalize_yoruba(text: str, level: Level = "full") -> str:
    text = unicodedata.normalize("NFD", text)
    text = text.replace(VERTICAL_LINE_BELOW, DOT_BELOW)
    if level in ("no_tone", "bare"):
        text = "".join(ch for ch in text if ch not in TONES)
    if level == "bare":
        text = text.replace(DOT_BELOW, "")
    return unicodedata.normalize("NFC", text)


def normalize_pidgin(text: str) -> str:
    return " ".join(PIDGIN_VARIANTS.get(word, word) for word in text.split())


def for_scoring(text: str, lang: str, level: Level = "full") -> str:
    """Lowercase, unify quotes, drop punctuation and extra spaces, then apply language rules."""
    text = unicodedata.normalize("NFC", text).translate(_QUOTES).lower()
    text = _SPACES.sub(" ", _PUNCT.sub(" ", text)).strip()
    if lang == "yor":
        return normalize_yoruba(text, level)
    if lang == "pcm":
        return normalize_pidgin(text)
    return text


def has_tone_marks(text: str) -> bool:
    """True if the text carries Yoruba tone marks. Untoned sources can't train a toned model."""
    return any(ch in TONES for ch in unicodedata.normalize("NFD", text))
