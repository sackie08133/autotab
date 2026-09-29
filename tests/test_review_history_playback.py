"""Regression tests for reversible editing, recovery, uncertainty, and audible review.

These tests use immutable projects and temporary storage rather than a real recording.
WAV samples are decoded to verify actual duration/pitch instead of merely checking
that a file exists. History tests include divergent edits, and autosave tests exercise
failure handling to ensure the last valid recovery snapshot survives a failed write.
"""

import io
import wave
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from autotab.autosave import available_drafts, save_draft
from autotab.domain import AudioNote, Contact, Note, Project, TimedContact
from autotab.fusion import fuse
from autotab.history import ProjectHistory
from autotab.playback import SAMPLE_RATE, synthesize
from autotab.review import alternatives, change_position, midi_pitch, replay_interval, review_reason
from autotab.storage import dumps, loads
from autotab.ui.calibration import clicked_calibration
from autotab.ui.editor import note_rows, parse_rows


def test_undo_redo_and_divergent_edit():
    """Undo restores all metadata, redo restores edits, and new edits drop the old branch."""
    original = Project("Test", 100, (Note(1, 0, 0, 1),))
    history = ProjectHistory(original)
    changed = replace(original, bpm=140, notes=(Note(2, 5, 0.5, 2),))
    assert history.commit(changed)
    assert not history.commit(changed)
    assert history.undo() == original
    assert history.redo() == changed
    history.undo()
    history.commit(replace(original, title="Different"))
    assert not history.can_redo
    assert history.redo().title == "Different"


def test_history_retention_is_bounded():
    """A long editing session retains the newest snapshots without unbounded growth."""
    history = ProjectHistory(Project("0", 100), limit=3)
    for number in range(1, 6):
        history.commit(Project(str(number), 100))
    assert history.undo().title == "4"
    assert history.undo().title == "3"
    assert history.undo().title == "3"


def test_autosave_sessions_do_not_overwrite_each_other(tmp_path):
    """Independent tabs get independent recovery files and valid data round-trips."""
    first, second = str(uuid4()), str(uuid4())
    project = Project("One", 100, (Note(1, 3, 0, 1, "audio", "Check position"),))
    path = save_draft(project, first, tmp_path)
    save_draft(Project("Two", 90), second, tmp_path)
    assert loads(path.read_text()) == project
    assert {p.title for _, p in available_drafts(tmp_path)} == {"One", "Two"}
    path.write_text("invalid JSON")
    assert [p.title for _, p in available_drafts(tmp_path)] == ["Two"]


def test_failed_autosave_preserves_last_good_file(tmp_path, monkeypatch):
    """Failure before atomic replacement leaves the old snapshot intact and cleans temp files."""
    session = str(uuid4())
    original = Project("Original", 100)
    path = save_draft(original, session, tmp_path)

    def fail_replace(*args):
        """Simulate a filesystem refusing the final rename."""
        raise OSError("disk unavailable")

    monkeypatch.setattr(type(path), "replace", fail_replace)
    with pytest.raises(OSError):
        save_draft(Project("Changed", 110), session, tmp_path)
    assert loads(path.read_text()) == original
    assert not list(tmp_path.glob("*.tmp"))


def test_autosave_rejects_path_traversal(tmp_path):
    """Only UUID session names can select a recovery path."""
    with pytest.raises(ValueError):
        save_draft(Project("Bad", 100), "../outside", tmp_path)


@pytest.mark.parametrize(
    "instrument,capo,string,fret",
    [
        ("guitar", 0, 2, 5),
        ("guitar", 2, 2, 5),
        ("bass", 0, 4, 10),
    ],
)
def test_alternatives_preserve_pitch_and_mark_manual(instrument, capo, string, fret):
    """Suggested fingerings are equivalent under the actual tuning/capo, including bass."""
    note = Note(string, fret, 0, 1, "audio")
    project = Project("Alternatives", 100, (note,), instrument=instrument, capo=capo)
    choices = alternatives(note, project)
    assert choices
    for candidate in choices:
        changed = change_position(project, 0, *candidate)
        assert midi_pitch(changed.notes[0], changed) == midi_pitch(note, project)
        assert not review_reason(changed.notes[0])
        assert changed.notes[0].source == "manual"


def test_alternative_does_not_overwrite_existing_note():
    """A same-pitch suggestion cannot occupy another simultaneous note's string."""
    project = Project("Chord", 100, (Note(2, 5, 0, 1), Note(1, 3, 0, 1)))
    with pytest.raises(ValueError, match="already starts"):
        change_position(project, 0, 1, 0)


def test_uncertainty_distinguishes_conflict_and_weak_audio():
    """Contradictory vision and weak periodicity both survive in the editable project."""
    result = fuse(
        (AudioNote(64, 0, 1, 0.85),), (TimedContact(Contact(6, 1), 0, 1),), Project("Conflict", 100)
    )
    note = result.notes[0]
    assert "general hand placement" in review_reason(note)
    assert "Weak audio" in review_reason(note)
    project = Project("Reasons", 100, (note,))
    assert loads(dumps(project)).notes == project.notes
    assert parse_rows(note_rows(project.notes), project.notes) == project.notes
    edited = note_rows(project.notes)
    edited[0]["Fret"] = 1
    assert not review_reason(parse_rows(edited, project.notes)[0])


def test_subsecond_replay_uses_offset_and_bounds():
    """Fast notes retain fractional seconds and a selected note must be in the recording."""
    project = Project("Replay", 120, beat_offset_seconds=1)
    start, end = replay_interval(Note(1, 2, 0.5, 0.25), project, 1.5)
    assert start == pytest.approx(0.95)
    assert end == 1.5
    with pytest.raises(ValueError, match="beyond"):
        replay_interval(Note(1, 2, 4, 1), project, 2)


@pytest.mark.parametrize(
    "instrument,capo,string,expected",
    [
        ("guitar", 0, 1, 329.63),
        ("guitar", 2, 1, 369.99),
        ("bass", 0, 4, 41.2),
    ],
)
def test_playback_wav_has_correct_pitch_and_duration(instrument, capo, string, expected):
    """Rendered audio reflects tempo, instrument tuning, and capo transposition."""
    project = Project("Sound", 120, (Note(string, 0, 0, 4),), instrument=instrument, capo=capo)
    with wave.open(io.BytesIO(synthesize(project, 0, 2))) as wav:
        assert wav.getnframes() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    spectrum = np.abs(np.fft.rfft(pcm.astype(float)))
    strongest = np.fft.rfftfreq(len(pcm), 1 / SAMPLE_RATE)[np.argmax(spectrum)]
    assert strongest == pytest.approx(expected, abs=1)


def test_loop_clips_sustained_notes_and_rejects_unbounded_output():
    """A note held across loop start remains audible; excessive/invalid ranges are refused."""
    project = Project("Sustain", 60, (Note(1, 0, 0, 4),))
    with wave.open(io.BytesIO(synthesize(project, 1, 2))) as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.max(np.abs(pcm)) > 1000
    for start, end in [(1, 1), (2, 1), (0, 121)]:
        with pytest.raises(ValueError):
            synthesize(project, start, end)


def test_clicked_calibration_validates_order_and_normalized_coordinates():
    """Browser clicks cross the same geometric validation boundary as manual coordinates."""
    value = {"corners": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]}
    assert clicked_calibration(value, 0, 12).corners[0] == (0.1, 0.2)
    for invalid in [{}, {"corners": [[0, 0]]}, {"corners": [[0, 0], [1, 1], [1, 0], [0, 1]]}]:
        with pytest.raises(ValueError):
            clicked_calibration(invalid, 0, 12)
