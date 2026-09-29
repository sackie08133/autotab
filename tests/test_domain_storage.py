"""Protect project invariants and portable saves without requiring a video or model.

These tests cover malformed user inputs and backward-compatible defaults. They
exercise observable behavior rather than the internal structure of dataclasses.
"""

import json

import pytest

from autotab.domain import Calibration, Note, Project
from autotab.storage import dumps, loads


@pytest.mark.parametrize("bpm", [0, -1, 401, float("nan"), float("inf"), True])
def test_invalid_tempo_rejected(bpm):
    """Invalid or implicit boolean tempo must never reach timing calculations."""
    with pytest.raises(ValueError):
        Project("Bad tempo", bpm)


def test_round_trip_bass_capo_and_calibration():
    """Saving/loading preserves musical edits, capo, tuning, and corner coordinates."""
    calibration = Calibration(((0.1, 0.2), (0.9, 0.3), (0.8, 0.8), (0.1, 0.7)))
    project = Project(
        "Bass test",
        95,
        (Note(4, 2, 1.5, 0.5, "matched"),),
        calibration=calibration,
        instrument="bass",
        capo=2,
    )
    assert loads(dumps(project)) == project


@pytest.mark.parametrize("payload", ["[]", "{}", '{"schema_version": 99}', "broken"])
def test_invalid_project_rejected(payload):
    """Malformed/unsupported project files produce recoverable validation errors."""
    with pytest.raises(ValueError):
        loads(payload)


def test_older_project_defaults_to_guitar_without_capo():
    """New optional metadata does not break version-one projects lacking those fields."""
    data = json.loads(dumps(Project("Old", 120)))
    del data["capo"]
    del data["instrument"]
    assert loads(json.dumps(data)).capo == 0
    assert loads(json.dumps(data)).instrument == "guitar"


def test_bass_rejects_fifth_string_and_capo_checks_physical_range():
    """Instrument-level constraints are checked after individual notes are valid."""
    with pytest.raises(ValueError, match="string"):
        Project("Bad bass", 100, (Note(5, 1, 0, 1),), instrument="bass")
    with pytest.raises(ValueError, match="physical fret"):
        Project("Bad capo", 100, (Note(1, 24, 0, 1),), capo=2)


def test_crossed_calibration_rejected():
    """A bow-tie polygon cannot define a usable perspective transform."""
    with pytest.raises(ValueError, match="convex"):
        Calibration(((0, 0), (1, 1), (1, 0), (0, 1)))
