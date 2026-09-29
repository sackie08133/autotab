"""Conversion between editable table rows and validated domain notes.

This module does not import Streamlit, making the boundary independently
testable. It prevents fractional string/fret values and partially filled rows
from becoming corrupt project data. Source labels describe analysis provenance;
edits to any musical field change a row's source to 'manual'.
"""

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

from autotab.domain import Note


def note_rows(notes: tuple[Note, ...]) -> list[dict[str, Any]]:
    """Create fresh table records with human-facing column names and beat units."""
    return [
        {
            "String": n.string,
            "Fret": n.fret,
            "Beat": n.beat,
            "Duration": n.duration,
            "Source": n.source,
        }
        for n in notes
    ]


def parse_rows(rows: Iterable[Mapping[str, Any]], original: tuple[Note, ...]) -> tuple[Note, ...]:
    """Validate edited records and return an immutable score, or a row-specific error.

    Empty records are ignored. Incomplete records are rejected so users can finish
    entering them before exporting. Musical values matching an original event keep
    its provenance; changed/new events are marked manual. Duplicate start positions
    on one string are invalid because a string cannot play two frets simultaneously.
    """
    provenance = {(n.string, n.fret, n.beat, n.duration): n for n in original}
    notes = []
    occupied = set()
    for index, row in enumerate(rows, start=1):
        values = [row.get(key) for key in ("String", "Fret", "Beat", "Duration")]
        if all(value is None for value in values):
            continue
        try:
            numeric = [float(value) for value in values]
            if not all(isfinite(value) for value in numeric):
                raise ValueError("Fill every musical field with a finite number.")
            string, fret, beat, duration = numeric
            if not string.is_integer() or not fret.is_integer():
                raise ValueError("String and fret must be whole numbers.")
            key = (int(string), int(fret), beat, duration)
            note = provenance.get(key) or Note(*key)
            if (note.string, note.beat) in occupied:
                raise ValueError("Two notes cannot start together on the same string.")
            occupied.add((note.string, note.beat))
            notes.append(note)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Row {index}: {exc}") from exc
    return tuple(sorted(notes, key=lambda n: (n.beat, n.string)))
