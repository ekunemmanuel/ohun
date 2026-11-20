"""The languages Ohùn supports.

Adding a language (Hausa, Igbo, Nigerian English, ...) starts here: add an entry, then
teach `ohun.text` how to normalize it. Everything else keys off the ISO 639-3 code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # ISO 639-3, used everywhere in Ohùn (data folders, catalog, CLI)
    name: str
    whisper: str | None  # Whisper's language token, None if Whisper doesn't know it
    fleurs: str | None  # FLEURS config name, used for the shared benchmark
    tonal: bool  # tone marks carry meaning, so scoring reports with and without them
    active: bool  # being worked on now, as opposed to planned


LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in [
        Language("yor", "Yoruba", whisper="yo", fleurs="yo_ng", tonal=True, active=True),
        # Whisper has no Pidgin token; English is the closest and what fine-tunes use.
        Language("pcm", "Nigerian Pidgin", whisper="en", fleurs=None, tonal=False, active=True),
        Language("hau", "Hausa", whisper="ha", fleurs="ha_ng", tonal=True, active=False),
        Language("ibo", "Igbo", whisper=None, fleurs="ig_ng", tonal=True, active=False),
    ]
}


def get(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        known = ", ".join(LANGUAGES)
        raise ValueError(f"Unknown language {code!r}. Known: {known}") from None
