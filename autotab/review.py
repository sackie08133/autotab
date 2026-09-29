"""Pure musical helpers for interactive note review and source-video navigation.

Uncertainty is expressed as reasons, never an invented accuracy percentage. Alternate
positions preserve the sounding pitch under the selected tuning/capo. Replay timing
uses the same user BPM and offset as transcription, and a calibrated marker represents
the tab's proposed position rather than claiming a freshly detected finger location.
"""

from dataclasses import replace

from autotab.domain import INSTRUMENTS, Note, Project
from autotab.fusion import positions


def midi_pitch(note: Note, project: Project) -> int:
    """Convert a capo-relative tab location to its sounding MIDI pitch."""
    return INSTRUMENTS[project.instrument].open_midi[note.string - 1] + project.capo + note.fret


def pitch_name(note: Note, project: Project) -> str:
    """Name the sounding pitch so a valid alternative fingering cannot conceal transposition."""
    midi = midi_pitch(note, project)
    names = ("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B")
    return f"{names[midi % 12]}{midi // 12 - 1}"


def review_reason(note: Note) -> str:
    """Describe missing/conflicting evidence without treating matches as certainty."""
    if note.source == "manual":
        return ""
    if note.review_reason:
        return note.review_reason
    if note.source == "audio":
        return "String/fret position is not visually confirmed."
    if note.source == "vision":
        return "Finger position only; pitch and picking are not audio-confirmed."
    return ""


def alternatives(note: Note, project: Project) -> tuple[tuple[int, int], ...]:
    """List other playable positions for the same pitch, excluding the current one."""
    return tuple(
        p
        for p in positions(midi_pitch(note, project), project.instrument, project.capo)
        if p != (note.string, note.fret)
    )


def change_position(project: Project, index: int, string: int, fret: int) -> Project:
    """Apply a same-pitch alternative as a manual correction, rejecting onset conflicts."""
    original = project.notes[index]
    if (string, fret) not in alternatives(original, project):
        raise ValueError("Choose an alternative with the same sounding pitch.")
    if any(
        i != index and n.string == string and n.beat == original.beat
        for i, n in enumerate(project.notes)
    ):
        raise ValueError("Another note already starts on that string at this beat.")
    notes = list(project.notes)
    notes[index] = replace(original, string=string, fret=fret, source="manual", review_reason="")
    return replace(project, notes=tuple(sorted(notes, key=lambda n: (n.beat, n.string))))


def replay_interval(note: Note, project: Project, duration: float) -> tuple[float, float]:
    """Return a short source-video interval with context, clamped to recording bounds.

    A note outside the uploaded recording raises a useful error rather than seeking
    to a misleading frame. Beat-based replay follows manual timing corrections.
    """
    onset = project.beat_offset_seconds + note.beat * 60 / project.bpm
    if onset >= duration:
        raise ValueError("This note is beyond the uploaded video. Check BPM and first-beat time.")
    return max(0, onset - 0.3), min(duration, onset + note.duration * 60 / project.bpm + 0.3)
