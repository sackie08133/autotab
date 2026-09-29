"""Numerical signal and perspective tests without external recordings or model files.

Synthetic harmonic tones cover the guitar/bass pitch range, silence, and repeated
attacks. Perspective tests verify physical fret spacing, cropped fretboards, and
camera orientation. These fixtures verify mechanics, not real-world accuracy.
"""

import numpy as np
import pytest

from autotab.adapters.audio import SAMPLE_RATE, analyze_audio, estimate_pitch
from autotab.adapters.geometry import FretboardMapper
from autotab.domain import Calibration, Contact


def tone(midi: int, seconds: float = 0.7) -> np.ndarray:
    """Generate a deterministic fundamental-plus-harmonics tone at a known MIDI pitch."""
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    hz = 440 * 2 ** ((midi - 69) / 12)
    return (
        0.3 * np.sin(2 * np.pi * hz * t)
        + 0.15 * np.sin(4 * np.pi * hz * t)
        + 0.06 * np.sin(6 * np.pi * hz * t)
    ).astype(np.float32)


@pytest.mark.parametrize(
    "midi,window",
    [
        (28, 4096),
        (33, 4096),
        (40, 2048),
        (45, 2048),
        (52, 2048),
        (64, 2048),
        (76, 2048),
        (88, 2048),
    ],
)
def test_pitch_estimate_across_guitar_and_bass_range(midi, window):
    """YIN must recover each synthetic fundamental within a quarter semitone."""
    expected = 440 * 2 ** ((midi - 69) / 12)
    hz, quality = estimate_pitch(tone(midi)[:window], 39, 1400)
    assert abs(12 * np.log2(hz / expected)) < 0.25
    assert quality > 0.8


@pytest.mark.parametrize("instrument,midi", [("guitar", 40), ("guitar", 64), ("bass", 28)])
def test_audio_event_detects_tone_after_silence(instrument, midi):
    """A known delayed tone yields a plausible pitch interval, including bass low E."""
    silence = np.zeros(round(0.3 * SAMPLE_RATE), dtype=np.float32)
    notes = analyze_audio(np.concatenate((silence, tone(midi), silence)), instrument)
    assert notes
    longest = max(notes, key=lambda n: n.end_seconds - n.start_seconds)
    assert longest.midi == midi
    assert abs(longest.start_seconds - 0.3) < 0.12
    assert abs(longest.end_seconds - 1.0) < 0.12


def test_silence_does_not_produce_audio_notes():
    """Zero-energy audio must remain empty, including short clips."""
    assert analyze_audio(np.zeros(SAMPLE_RATE, dtype=np.float32)) == ()
    assert analyze_audio(np.zeros(100, dtype=np.float32)) == ()


def test_repeated_note_with_quiet_gap_is_split():
    """Two attacks at the same pitch separated by silence remain two editable events."""
    gap = np.zeros(round(0.15 * SAMPLE_RATE), dtype=np.float32)
    notes = analyze_audio(np.concatenate((tone(57, 0.4), gap, tone(57, 0.4))))
    assert len(notes) == 2
    assert [note.midi for note in notes] == [57, 57]


def test_repeated_pluck_without_full_silence_is_split():
    """An energy attack splits re-picking while the fundamental never fully stops."""
    pluck = tone(57, 0.5) * np.exp(-np.arange(round(SAMPLE_RATE * 0.5)) / SAMPLE_RATE * 8)
    notes = analyze_audio(np.concatenate((pluck, pluck)))
    assert len(notes) == 2
    assert all(note.midi == 57 for note in notes)


@pytest.mark.parametrize(
    "corners",
    [
        ((0.1, 0.2), (0.9, 0.3), (0.8, 0.8), (0.2, 0.7)),
        ((0.9, 0.7), (0.1, 0.8), (0.2, 0.2), (0.8, 0.3)),
    ],
)
def test_perspective_contact_and_rotation(corners):
    """Projection finds the same string/fret after perspective distortion or rotation."""
    mapper = FretboardMapper(Calibration(corners))
    along = (mapper.boundaries[4] + mapper.boundaries[5]) / 2
    x, y = mapper.image_point(along, 1 / 5)
    assert mapper.contact(x, y) == Contact(2, 5)
    assert mapper.contact(*mapper.image_point(along, 0.1)) is None
    assert mapper.contact(*mapper.image_point(1.1, 0)) is None


def test_cropped_bass_geometry_uses_four_strings():
    """A bass crop beginning at wire 3 still maps the following fret to fret 4."""
    mapper = FretboardMapper(Calibration(((0, 0), (1, 0), (1, 1), (0, 1)), 3, 12), 4)
    point = mapper.image_point(mapper.boundaries[1] / 2, 1)
    assert mapper.contact(*point) == Contact(4, 4)
