"""Scoring for speech to text.

WER (word error rate) is the headline number: lower is better, 0 is perfect. CER
(character error rate) is kinder to small spelling slips. For tonal languages we also
report WER with tones removed, so we can see how much of the error is tone alone.
"""

import unicodedata
from dataclasses import dataclass

import jiwer

from ohun import languages
from ohun.text import DOT_BELOW, Level, for_scoring, has_tone_marks


@dataclass
class Score:
    lang: str
    n: int
    wer: float
    cer: float
    wer_no_tone: float | None = None
    wer_bare: float | None = None

    def summary(self) -> str:
        parts = [f"{self.lang}  n={self.n}  WER {self.wer:.1%}  CER {self.cer:.1%}"]
        if self.wer_no_tone is not None:
            parts.append(f"WER without tones {self.wer_no_tone:.1%}")
        if self.wer_bare is not None:
            parts.append(f"WER without any marks {self.wer_bare:.1%}")
        return "  |  ".join(parts)


def reference_level(ref: str) -> Level:
    """The richest level a Yoruba reference supports.

    About 16% of FLEURS Yoruba test references carry no tone marks (some no under-dots
    either). Scoring those at "full" would punish a model for writing tones correctly, so
    each clip is scored at the level its own reference is written in.
    """
    if has_tone_marks(ref):
        return "full"
    return "no_tone" if DOT_BELOW in unicodedata.normalize("NFD", ref) else "bare"


def _clean(refs: list[str], hyps: list[str], lang: str, level) -> tuple[list[str], list[str]]:
    def lvl(ref: str) -> Level:
        if lang == "yor" and level == "full":
            return reference_level(ref)
        return level

    pairs = [(for_scoring(r, lang, lvl(r)), for_scoring(h, lang, lvl(r))) for r, h in zip(refs, hyps)]
    # jiwer can't score an empty reference; those clips carry no signal anyway.
    pairs = [(r, h) for r, h in pairs if r]
    return [r for r, _ in pairs], [h for _, h in pairs]


def score(references: list[str], hypotheses: list[str], lang: str) -> Score:
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    refs, hyps = _clean(references, hypotheses, lang, "full")
    if not refs:
        raise ValueError("Nothing to score: every reference is empty after normalization")
    result = Score(lang=lang, n=len(refs), wer=jiwer.wer(refs, hyps), cer=jiwer.cer(refs, hyps))
    languages.get(lang)  # fail early on an unknown code
    if lang == "yor":  # the only language with tone-level normalization so far
        result.wer_no_tone = jiwer.wer(*_clean(references, hypotheses, lang, "no_tone"))
        result.wer_bare = jiwer.wer(*_clean(references, hypotheses, lang, "bare"))
    return result
