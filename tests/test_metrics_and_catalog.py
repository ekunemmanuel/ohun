import pytest

from ohun import catalog
from ohun.cli import main
from ohun.metrics import score


def test_perfect_transcript_scores_zero():
    s = score(["Ẹ káàárọ̀ o"], ["ẹ káàárọ̀ o."], "yor")
    assert s.wer == 0 and s.cer == 0


def test_wrong_tone_counts_as_error_only_at_full_level():
    s = score(["ọkọ̀ náà dé"], ["ọkọ́ náà dé"], "yor")
    assert s.wer == pytest.approx(1 / 3)
    assert s.wer_no_tone == 0
    assert s.wer_bare == 0


def test_untoned_reference_does_not_punish_toned_output():
    # FLEURS has references typed without marks; a correct toned answer must still score 0
    # first reference has under-dots but no tones, second has no marks at all
    s = score(["Ẹ kaaarọ o", "MS je aarun"], ["Ẹ káàárọ̀ o", "MS jẹ́ aàrùn"], "yor")
    assert s.wer == 0


def test_toned_reference_still_counts_tone_errors():
    s = score(["ọkọ̀ náà dé", "MS je aarun"], ["ọkọ́ náà dé", "MS je aarun"], "yor")
    assert s.wer == pytest.approx(1 / 6)


def test_pidgin_has_no_tone_scores():
    s = score(["wetin dey happen"], ["wettin dey happen"], "pcm")
    assert s.wer == 0
    assert s.wer_no_tone is None


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        score(["a", "b"], ["a"], "pcm")


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        score(["a"], ["a"], "xyz")


def test_catalog_loads_and_every_entry_has_a_license():
    entries = catalog.load()
    assert {e.kind for e in entries} == {"dataset", "model"}
    assert all(e.license for e in entries)
    assert len({e.id for e in entries}) == len(entries), "duplicate ids"


def test_commercial_filter_drops_research_only_entries():
    entries = catalog.find(catalog.load(), lang="yor", task="stt", commercial_only=True)
    ids = {e.id for e in entries}
    assert "naijavoices" not in ids  # non-commercial
    assert "fleurs-yor" in ids
    assert "whisper-turbo" in ids  # models apply to every language


def test_cli_catalog_runs(capsys):
    main(["catalog", "--lang", "pcm", "--task", "tts"])
    out = capsys.readouterr().out
    assert "pidgin-1.0" in out and "bibletts-yor" not in out
