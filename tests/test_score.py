"""Behavioral checks for bar-based tablature and standard rhythmic engraving.

These tests parse exported SVG rather than comparing a brittle snapshot. They
verify source-note mapping, exact onset retention, barline ties, rhythm values,
beam grouping, string order for bass, and escaping of untrusted project titles.
Generated SVG remains standalone and uses no installed musical glyph font.
"""

import xml.etree.ElementTree as ET

import pytest

from autotab.domain import Note, Project
from autotab.score import bars, rhythm, to_svg


@pytest.mark.parametrize(
    "duration,name,hollow,stem,flags,dotted",
    [
        (4, "whole", True, False, 0, False),
        (2, "half", True, True, 0, False),
        (1, "quarter", False, True, 0, False),
        (0.5, "eighth", False, True, 1, False),
        (0.25, "sixteenth", False, True, 2, False),
        (0.125, "thirty-second", False, True, 3, False),
        (1.5, "dotted quarter", False, True, 0, True),
    ],
)
def test_standard_rhythm_shapes(duration, name, hollow, stem, flags, dotted):
    """Whole and half notes must differ from filled quarter/eighth note symbols."""
    value = rhythm(duration)
    assert (value.name, value.hollow, value.stem, value.flags, value.dotted) == (
        name,
        hollow,
        stem,
        flags,
        dotted,
    )


def test_nonstandard_durations_are_not_silently_changed_to_quarters():
    """Imprecise automatic durations stay visible until the user corrects them."""
    assert not rhythm(0.9).exact
    assert rhythm(0.9).code == "0.9b"


def test_barline_sustain_keeps_source_identity_and_does_not_create_new_attacks():
    """A note from beat 3 to 9 spans three bars, retaining its original score entry."""
    project = Project("Sustain", 100, (Note(3, 9, 3, 6),))
    measures = bars(project)
    assert [number for number, _ in measures] == [1, 2, 3]
    segments = [events[0] for _, events in measures]
    assert [(s.beat, s.duration, s.continued, s.continues) for s in segments] == [
        (3, 1, False, True),
        (4, 4, True, True),
        (8, 1, True, False),
    ]
    assert {s.index for s in segments} == {0}
    svg = to_svg(project)
    assert svg.count('class="tie"') == 4
    assert "(9)" in svg
    assert project.notes[0].duration == 6


def test_bar_numbers_preserve_setup_offset_and_empty_bars():
    """Leading setup is omitted, but internal rests and absolute landmarks survive."""
    measures = bars(Project("Gap", 100, (Note(3, 9, 4.75, 1), Note(3, 5, 12, 1))))
    assert [number for number, _ in measures] == [2, 3, 4]
    assert measures[1][1] == ()


def test_svg_keeps_onsets_and_strings_while_showing_only_bar_landmarks():
    """Each displayed fret traces to its exact source note, with no beat ruler."""
    project = Project("Exact", 100, (Note(3, 9, 0, 1), Note(2, 5, 0.13, 0.25)))
    root = ET.fromstring(to_svg(project))
    notes = [e for e in root.iter() if e.get("class") == "tab-note"]
    assert [
        (n.get("data-note-index"), n.get("data-beat"), n.get("data-string")) for n in notes
    ] == [("0", "0", "3"), ("1", "0.13", "2")]
    labels = [e.text for e in root.iter() if e.tag.endswith("}text")]
    assert "1" in labels
    assert "0.13" not in labels


def test_beams_group_short_notes_within_a_beat_but_not_across_rests():
    """Two eighths beam together; a later short note after silence gets its own flag."""
    notes = (Note(3, 9, 0, 0.5), Note(3, 7, 0.5, 0.5), Note(3, 5, 1.5, 0.25))
    root = ET.fromstring(to_svg(Project("Beams", 100, notes)))
    assert len([e for e in root.iter() if e.get("class") == "beam"]) == 1


def test_svg_escapes_title_and_supports_bass_chord_durations():
    """User text stays text, while simultaneous notes can retain different rhythms."""
    project = Project(
        '<script>alert("x")</script>', 80, (Note(1, 3, 0, 1), Note(4, 5, 0, 2)), instrument="bass"
    )
    svg = to_svg(project)
    root = ET.fromstring(svg)
    assert not any(e.tag.endswith("}script") for e in root.iter())
    labels = [e.text for e in root.iter() if e.tag.endswith("}text")]
    assert "G" in labels and "E" in labels and "B" not in labels
    marks = [e for e in root.iter() if e.get("class") == "rhythm"]
    # The first two marks belong to the chord; the remaining marks are the legend.
    assert {e.get("data-duration") for e in marks[:2]} == {"quarter", "half"}


def test_preview_rejects_impossible_duplicate_string_onsets():
    """Neither conflicting fret may be silently overwritten by an SVG group."""
    with pytest.raises(ValueError, match="exact beat"):
        to_svg(Project("Conflict", 100, (Note(3, 9, 0, 1), Note(3, 7, 0, 1))))
