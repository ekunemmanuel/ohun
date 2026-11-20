import unicodedata

from ohun.text import for_scoring, has_tone_marks, normalize_yoruba


def test_precomposed_and_decomposed_yoruba_are_equal():
    precomposed = unicodedata.normalize("NFC", "ọkọ̀")
    decomposed = unicodedata.normalize("NFD", "ọkọ̀")
    assert precomposed != decomposed
    assert normalize_yoruba(precomposed) == normalize_yoruba(decomposed)


def test_vertical_line_below_becomes_dot_below():
    assert normalize_yoruba("e̩") == "ẹ"
    assert normalize_yoruba("s̩e̩") == "ṣẹ"


def test_tones_survive_full_level():
    # hoe, vehicle and husband must stay three different words
    words = {normalize_yoruba(w) for w in ["ọkọ́", "ọkọ̀", "ọkọ"]}
    assert len(words) == 3


def test_no_tone_keeps_underdots():
    assert normalize_yoruba("Ọkọ̀ náà", "no_tone") == "Ọkọ naa"


def test_bare_strips_everything():
    assert normalize_yoruba("Ọkọ̀ náà ṣẹ́", "bare") == "Oko naa se"


def test_scoring_lowercases_and_drops_punctuation():
    assert for_scoring("  Ẹ ṣé o!  “Ó dára.” ", "yor") == "ẹ ṣé o ó dára"


def test_scoring_keeps_word_internal_apostrophe():
    assert for_scoring("I no fit go o, na im dey there!", "pcm") == "i no fit go o na im dey there"
    assert for_scoring("Wetin don’t happen?", "pcm") == "wetin don't happen"


def test_pidgin_variants_collapse():
    assert for_scoring("Wettin dis pickin talk", "pcm") == "wetin this pikin talk"


def test_bbc_pidgin_spelling_matches_english_spelling():
    bbc = "plenti pipo don show for di mountain"
    english = "Plenty people don show for the mountain."
    assert for_scoring(bbc, "pcm") == for_scoring(english, "pcm")


def test_dem_is_left_alone():
    assert for_scoring("dem don go", "pcm") == "dem don go"


def test_has_tone_marks():
    assert has_tone_marks("Ẹ káàárọ̀")
    assert not has_tone_marks("Ẹ kaaro")
