import pytest

from sharp import labels

# Every label string that appears in Harmonix Set v1.2 segments/*.txt
HARMONIX_RAW_LABELS = [
    'altchorus', 'bigoutro', 'bre', 'break', 'break1', 'break2', 'break3', 'breakdown', 'breakdown2',
    'bridge', 'bridge1', 'bridge2', 'bridge3', 'build', 'chorus', 'chorus1', 'chorus2', 'chorus3',
    'chorus_instrumental', 'chorushalf', 'chorusinst', 'choruspart', 'drumroll', 'end', 'fadein', 'fast',
    'gtr', 'gtr2', 'gtrbreak', 'guitar', 'guitarsolo', 'inst', 'inst2', 'instbridge', 'instchorus',
    'instintro', 'instrumental', 'instrumental2', 'instrumental3', 'instrumentalverse', 'intchorus',
    'intro', 'intro2', 'intro3', 'intro4', 'intro5', 'intro6', 'intro7', 'intro8', 'introchorus',
    'intropt2', 'introverse', 'mainriff', 'mainriff2', 'miniverse', 'oddriff', 'opening', 'outro',
    'outro1', 'outro2', 'outro3', 'outroa', 'postchorus', 'postchorus2', 'postverse', 'prechorus',
    'prechorus2', 'prechorus3', 'prechorus5', 'preverse', 'quiet', 'quietchorus', 'raps', 'refrain',
    'rhythmlessintro', 'saxobeat', 'section', 'section1', 'section10', 'section11', 'section12',
    'section13', 'section14', 'section15', 'section16', 'section17', 'section2', 'section3', 'section4',
    'section5', 'section6', 'section7', 'section8', 'section9', 'silence', 'slow', 'slow2', 'slowverse',
    'solo', 'solo2', 'solo3', 'stutter', 'synth', 'transition', 'transition1', 'transition2',
    'transition2a', 'transition3', 'verse', 'verse1', 'verse10', 'verse11', 'verse1a', 'verse2', 'verse3',
    'verse4', 'verse5', 'verse6', 'verse7', 'verse8', 'verse9', 'verse_slow', 'verseinst', 'versepart',
    'vocaloutro', 'worstthingever',
]


def test_classes_map_to_themselves():
    for c in labels.CLASSES:
        assert labels.to_class(c) == c
        assert labels.to_index(c) == labels.CLASS_TO_INDEX[c]


@pytest.mark.parametrize("raw,expected", [
    ("verse2", "verse"), ("verse1a", "verse"), ("Verse 3", "verse"), ("pre-chorus", "prechorus"),
    ("Chorus", "chorus"), ("chorus_instrumental", "chorus"), ("postchorus2", "postchorus"),
    ("inst", "instrumental"), ("solo3", "instrumental"), ("gtr2", "instrumental"),
    ("transition2a", "break"), ("breakdown", "break"), ("outroa", "outro"), ("intropt2", "intro"),
    ("instbridge", "bridge"), ("build", "prechorus"), ("silence", "silence"),
    ("section10", None), ("end", None), ("worstthingever", None),
])
def test_specific_mappings(raw, expected):
    assert labels.to_class(raw) == expected


def test_every_harmonix_label_is_handled():
    unmapped = {r for r in HARMONIX_RAW_LABELS if labels.to_class(r) is None}
    assert all(r.startswith("section") for r in unmapped - {"end", "fast", "worstthingever"})
    assert len(HARMONIX_RAW_LABELS) - len(unmapped) >= 100


def test_ignore_index_for_unmapped():
    assert labels.to_index("section") == labels.IGNORE_INDEX


def test_every_class_has_colour_and_display_name():
    for c in labels.CLASSES:
        assert labels.color_for(c).startswith("#")
        assert labels.display_name(c)
