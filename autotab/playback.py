"""Bounded synthesized tab playback for checking pitch, rhythm, and edited durations.

This is a lightweight reference instrument, not audio reconstruction. A decaying
harmonic tone is generated for each event and mixed at its beat position, with the
chosen tuning/capo determining pitch. Selecting a beat range clips sustained notes
correctly and makes looping independent of a project's first-beat video offset.
Output is ordinary mono 16-bit WAV, playable/downloadable without extra services.
"""

import io
import wave

import numpy as np

from autotab.domain import Project, finite_number
from autotab.review import midi_pitch

SAMPLE_RATE = 22050
MAX_SECONDS = 120


def synthesize(project: Project, start_beat: float, end_beat: float) -> bytes:
    """Render a beat interval to WAV, limiting output to two minutes per selection.

    Notes beginning before the interval retain their envelope/phase at the boundary.
    Chords mix before normalization, preventing clipping. The audio player owns
    looping; no repeated copies are allocated here. Empty ranges render silence.
    """
    finite_number(start_beat, "Playback start")
    finite_number(end_beat, "Playback end")
    seconds_per_beat = 60 / project.bpm
    duration = (end_beat - start_beat) * seconds_per_beat
    if duration <= 0 or duration > MAX_SECONDS:
        raise ValueError("Select a playback range longer than zero and at most 120 seconds.")
    samples = np.zeros(max(1, round(duration * SAMPLE_RATE)), dtype=np.float64)
    for note in project.notes:
        begin = max(start_beat, note.beat)
        end = min(end_beat, note.beat + note.duration)
        if begin >= end:
            continue
        first = round((begin - start_beat) * seconds_per_beat * SAMPLE_RATE)
        last = min(len(samples), round((end - start_beat) * seconds_per_beat * SAMPLE_RATE))
        t = np.arange(last - first) / SAMPLE_RATE + (begin - note.beat) * seconds_per_beat
        note_seconds = note.duration * seconds_per_beat
        hz = 440 * 2 ** ((midi_pitch(note, project) - 69) / 12)
        envelope = np.minimum(1, t / 0.008) * np.minimum(1, np.maximum(0, note_seconds - t) / 0.03)
        envelope *= 0.25 + 0.75 * np.exp(-t * 3)
        tone = (
            np.sin(2 * np.pi * hz * t)
            + 0.3 * np.sin(4 * np.pi * hz * t)
            + 0.1 * np.sin(6 * np.pi * hz * t)
        )
        samples[first:last] += envelope * tone
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples *= 0.85 / peak
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return output.getvalue()
