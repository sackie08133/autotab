"""Exercise manual corrections, provenance, and user-facing export semantics.

Editing must never silently truncate fractional frets or omit conflicting notes.
Text and CSV exports are checked as user-visible data, not implementation details.
"""

import csv
import io

import pytest

from autotab.domain import Note, Project
from autotab.export import to_ascii, to_csv
from autotab.ui.editor import note_rows, parse_rows


def test_edit_changes_provenance_and_allows_adding_deleting():
    """Corrections become manual, unchanged rows retain their evidence source."""
    original = (Note(2, 5, 0, 1, "matched"), Note(1, 3, 1, 1, "audio"))
    rows = note_rows(original)
    rows[0]["Fret"] = 7
    notes = parse_rows(rows, original)
    assert notes[0].source == "manual"
    assert notes[1].source == "audio"
    assert parse_rows([], original) == ()
    assert (
        parse_rows([{"String": 6, "Fret": 0, "Beat": 0, "Duration": 1}], ())[0].source == "manual"
    )


@pytest.mark.parametrize(
    "field,value", [("Fret", 2.5), ("String", 7), ("Beat", -1), ("Duration", 0), ("Fret", None)]
)
def test_invalid_edit_reports_row(field, value):
    """The user gets a useful row-specific message instead of a corrupt saved score."""
    row = note_rows((Note(1, 0, 0, 1),))[0]
    row[field] = value
    with pytest.raises(ValueError, match="Row 1"):
        parse_rows([row], ())


def test_ascii_preserves_chord_alignment_and_two_digit_frets():
    """Different fret widths must not shift simultaneous notes into different columns."""
    tab = to_ascii(Project("Chord", 100, (Note(1, 12, 0, 1), Note(2, 3, 0, 1))))
    assert "e| 12--" in tab
    assert "B| 3---" in tab


def test_ascii_preserves_two_distinct_onsets_inside_one_grid_cell():
    """Rendering must not round an edited onset again or merge two distinct notes."""
    tab = to_ascii(Project("Close notes", 100, (Note(1, 1, 0, 1), Note(1, 2, 0.01, 1))))
    assert "Bar 1" in tab
    assert "0.01" not in tab
    assert "e| 1---2----" in tab


def test_ascii_rejects_actual_simultaneous_frets_on_one_string():
    """A real contradictory list is reported rather than hiding one of its entries."""
    with pytest.raises(ValueError, match="exact beat"):
        to_ascii(Project("Conflict", 100, (Note(1, 1, 0, 1), Note(1, 2, 0, 1))))


def test_csv_seconds_use_user_bpm_and_offset():
    """A beat-two note at 120 BPM with a three-second offset begins at second four."""
    project = Project("Timing", 120, (Note(4, 5, 2, 0.5),), beat_offset_seconds=3)
    row = list(csv.DictReader(io.StringIO(to_csv(project))))[0]
    assert float(row["start_seconds"]) == 4
    assert float(row["duration_beats"]) == 0.5
