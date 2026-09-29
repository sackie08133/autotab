"""Behavioral tests for temporal evidence, beat offsets, and service orchestration.

Fake adapters make these tests independent of video codecs and ML model weights.
The scenarios exercise dropouts, chord contacts, frame ordering, and the important
distinction between absent audio analysis and audio analysis finding no notes.
"""

import pytest

from autotab.domain import AudioNote, Contact, Project, TimedContact
from autotab.pipeline import analyze
from autotab.ports import VideoFrame
from autotab.transcription import ContactTracker, quantize


def test_tracker_filters_flicker_and_bridges_dropout():
    """A stable contact survives one missing frame; one-frame noise never becomes a note."""
    tracker = ContactTracker(release_seconds=0.15)
    good, noise = Contact(2, 3), Contact(3, 5)
    tracker.update(0, (good, noise))
    tracker.update(0.05, (good,))
    tracker.update(0.10, ())
    tracker.update(0.15, (good,))
    result = tracker.finish(0.20)
    assert len(result) == 1
    assert result[0] == TimedContact(good, 0, 0.20)


def test_higher_fret_replaces_lower_on_same_string():
    """Two fingers on one string cannot create simultaneous sounding fret guesses."""
    tracker = ContactTracker(min_observations=1)
    tracker.update(0, (Contact(1, 2),))
    tracker.update(0.1, (Contact(1, 2), Contact(1, 5)))
    result = tracker.finish(0.2)
    assert [event.contact.fret for event in result] == [2, 5]
    assert result[0].end_seconds == result[1].start_seconds


def test_timestamp_regression_rejected():
    """A broken decoder cannot silently corrupt durations by reordering frames."""
    tracker = ContactTracker()
    tracker.update(1, ())
    with pytest.raises(ValueError, match="increase"):
        tracker.update(0.5, ())


def test_user_bpm_offset_and_short_duration():
    """Notes before the first beat are dropped; brief later notes get one grid cell."""
    events = (TimedContact(Contact(1, 3), 0, 0.5), TimedContact(Contact(2, 4), 1.26, 1.30))
    notes = quantize(events, Project("Timing", 120, beat_offset_seconds=1))
    assert len(notes) == 1
    assert notes[0].beat == 0.5
    assert notes[0].duration == 0.25


class FakeSource:
    """Generate a short known frame sequence without allocating image pixels."""

    def frames(self):
        """Yield eleven chronological samples spanning one second."""
        return (VideoFrame(i / 10, None) for i in range(11))


class FakeDetector:
    """Supply a constant E4 contact at string 2, fret 5."""

    def detect(self, frame):
        """Return evidence independent of the fake frame's opaque image value."""
        return (Contact(2, 5),)


def test_pipeline_cross_matches_audio_and_keeps_input_immutable():
    """Audio onset survives orchestration and the matching visual position is chosen."""
    settings = Project("Integration", 120)
    result = analyze(
        FakeSource(), FakeDetector(), settings, 0.1, audio_notes=(AudioNote(64, 0.25, 0.75, 0.9),)
    )
    assert result.project.notes[0].source == "matched"
    assert result.project.notes[0].string == 2
    assert result.project.notes[0].beat == 0.5
    assert settings.notes == ()
    assert result.frames_processed == 11


def test_silent_audio_is_not_replaced_with_visual_notes():
    """An analyzed silent soundtrack must produce an empty audiovisual draft."""
    result = analyze(FakeSource(), FakeDetector(), Project("Silent", 120), 0.1, audio_notes=())
    assert result.project.notes == ()
