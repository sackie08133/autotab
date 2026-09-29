"""Resolve acoustic pitch ambiguity using simultaneous visual fretboard evidence.

The same pitch can appear on several strings. This pure module enumerates
playable positions under the selected standard tuning. Audio exclusively supplies
pitch and note events; the general hand region selects plausible fingerings for
that pitch. Precise contact estimates only help distinguish nearby alternatives.
Conflicts remain visible as audio-only guesses instead of being hidden.

This is a transparent baseline, not a learned multimodal network. Its interfaces
allow better pitch or contact models later without changing saved project files.
"""

from dataclasses import dataclass
from math import exp, floor
from statistics import median

from autotab.domain import (
    INSTRUMENTS,
    AudioNote,
    Contact,
    Note,
    Project,
    TimedContact,
    VisualSample,
)


@dataclass(frozen=True)
class FusionResult:
    """Quantized notes and counts describing matching, fallback, and rejected pitches."""

    notes: tuple[Note, ...]
    matched: int
    unconfirmed: int
    out_of_range: int


def positions(midi: int, instrument: str = "guitar", capo: int = 0) -> tuple[tuple[int, int], ...]:
    """Enumerate all string/fret locations for a pitch within frets zero through 24.

    String numbering begins at the highest string for both guitar and bass. An empty
    result means the pitch is outside this instrument's modeled playable range.
    """
    return tuple(
        (string, midi - pitch - capo)
        for string, pitch in enumerate(INSTRUMENTS[instrument].open_midi, start=1)
        if 0 <= midi - pitch - capo <= 24 - capo
    )


def _position_evidence(
    candidate: tuple[int, int],
    event: AudioNote,
    end: float,
    contacts: tuple[TimedContact, ...],
    samples: tuple[VisualSample, ...],
    tolerance: float,
) -> float:
    """Score a same-pitch location around the attack, before the hand prepares a later note.

    Continuous positions tolerate up to 0.85 string spacings and 0.35 fret widths
    beyond the target compartment, with rapidly decaying support. This preserves
    near-string landmarks without treating all nearby pitches as equivalent. Hard
    contacts are a secondary signal when richer observations are available.
    """
    string, fret = candidate
    if fret == 0:
        return 0.0
    begin = max(0, event.start_seconds - tolerance)
    span = max(0.001, end - begin)
    exact = min(
        1.0,
        sum(
            max(0.0, min(end, v.end_seconds) - max(begin, v.start_seconds))
            for v in contacts
            if v.contact == Contact(string, fret)
        )
        / span,
    )
    if not samples:
        return exact
    frame_scores = []
    for sample in samples:
        best = 0.0
        for tip in sample.fingertips:
            across = abs(tip.string_position - string)
            along = max(fret - 1 - tip.fret_position, tip.fret_position - fret, 0)
            if across <= 0.85 and along <= 0.35:
                best = max(best, exp(-0.5 * (across / 0.45) ** 2 - 0.5 * (along / 0.22) ** 2))
        frame_scores.append(best)
    return 0.8 * sum(frame_scores) / len(frame_scores) + 0.2 * exact


def fuse(
    audio: tuple[AudioNote, ...],
    visual: tuple[TimedContact, ...],
    settings: Project,
    alignment_tolerance: float = 0.10,
    visual_samples: tuple[VisualSample, ...] = (),
) -> FusionResult:
    """Audio controls onset/pitch; overlapping, pitch-consistent vision selects position.

    The median hand position first narrows the audio-compatible alternatives to
    the closest neck region. A one-fret distance allowance avoids overreacting to
    small landmark errors; precise string/contact scores distinguish alternatives
    within that allowance. Without visual evidence, continuity supplies a guess.
    Audio pitch and event count never change to satisfy a visual observation.
    """
    notes: list[Note] = []
    previous_fret = 0
    outside = 0
    ordered_audio = sorted(audio, key=lambda note: note.start_seconds)
    for index, event in enumerate(ordered_audio):
        candidates = positions(event.midi, settings.instrument, settings.capo)
        if not candidates:
            outside += 1
            continue
        start = (event.start_seconds - settings.beat_offset_seconds) * settings.bpm / 60
        end = (event.end_seconds - settings.beat_offset_seconds) * settings.bpm / 60
        if end <= 0:
            continue
        evidence_end = min(event.end_seconds, event.start_seconds + 0.25)
        if index + 1 < len(ordered_audio):
            next_onset = ordered_audio[index + 1].start_seconds
            if next_onset > event.start_seconds:
                evidence_end = min(evidence_end, next_onset)
        samples = tuple(
            sample
            for sample in visual_samples
            if event.start_seconds - alignment_tolerance <= sample.seconds < evidence_end
            and sample.fingertips
        )
        scores = {
            candidate: _position_evidence(
                candidate, event, evidence_end, visual, samples, alignment_tolerance
            )
            for candidate in candidates
        }
        # Estimate the hand's general neck position independently of which string
        # each fingertip appears to touch. First take a median within each frame,
        # then across frames: one stray finger or a frame with more visible fingers
        # should not pull the whole performance toward another neck position.
        visible_frets = [
            median(tip.fret_position + 0.5 for tip in sample.fingertips) for sample in samples
        ]
        if not visible_frets:
            visible_frets = [
                v.contact.fret
                for v in visual
                if v.start_seconds < evidence_end
                and v.end_seconds > event.start_seconds - alignment_tolerance
            ]
        hand_fret = median(visible_frets) if visible_frets else previous_fret
        nearby = candidates
        if visible_frets:
            nearest_distance = min(abs(fret - hand_fret) for _, fret in candidates)
            # General hand placement is the primary visual cue. Exact contacts
            # cannot pull a note to a distant same-pitch position. Keep candidates
            # within one fret of the best distance for uncertain region boundaries.
            nearby = tuple(p for p in candidates if abs(p[1] - hand_fret) <= nearest_distance + 1.0)
        string, fret = max(
            nearby,
            key=lambda p: (
                scores[p],
                -abs(p[1] - hand_fret),
                -abs(p[1] - previous_fret),
                -p[1],
                -p[0],
            ),
        )
        best_score = scores[(string, fret)]
        runner_up = max((score for p, score in scores.items() if p != (string, fret)), default=0)
        confirmed = best_score >= 0.3 and best_score - runner_up >= 0.12
        source = "matched" if confirmed else "audio"
        reason = ""
        if source == "audio":
            if best_score > 0:
                reason = "Video suggests this position, but finger/string evidence is ambiguous."
            elif visible_frets:
                reason = (
                    "Fingering inferred from general hand placement; exact string is unconfirmed."
                )
            else:
                reason = "No usable visual evidence; fingering is only a pitch-compatible guess."
        if event.quality < 0.9:
            reason = (reason + " Weak audio periodicity; verify the pitch.").strip()
        grid = settings.subdivisions
        beat = floor(max(0, start) * grid + 0.5) / grid
        end_beat = floor(end * grid + 0.5) / grid
        # Do not erase rapid repeated notes when they round to the same grid cell.
        # Keep their measured beat if the proposed cell is already occupied.
        if any(n.string == string and n.beat == beat for n in notes):
            beat = max(0, start)
        notes.append(Note(string, fret, beat, max(1 / grid, end_beat - beat), source, reason))
        previous_fret = fret
    ordered = tuple(sorted(notes, key=lambda note: (note.beat, note.string)))
    matched = sum(note.source == "matched" for note in ordered)
    return FusionResult(ordered, matched, len(ordered) - matched, outside)
