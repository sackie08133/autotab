"""Coordinate visual evidence and optional audio observations into a draft score.

This application service knows only the adapter protocols and domain objects.
It does not decode files, initialize models, or render widgets. The caller owns
those resources and can surface progress in a UI or CLI. Returning a new Project
keeps a user's current edited draft intact if analysis raises an exception.

audio_notes=None explicitly selects vision-only behavior; an empty audio tuple
means audio was analyzed but found no notes and must not trigger a silent fallback.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from autotab.domain import AudioNote, Project, VisualSample, finite_number
from autotab.fusion import fuse
from autotab.ports import ContactDetector, FingertipDetector, FrameSource
from autotab.transcription import ContactTracker, quantize


@dataclass(frozen=True)
class AnalysisResult:
    """A draft plus diagnostics for showing how much supporting evidence was found."""

    project: Project
    frames_processed: int
    frames_with_contacts: int
    matched_notes: int = 0
    unconfirmed_notes: int = 0
    out_of_range_notes: int = 0
    visual_samples: tuple[VisualSample, ...] = ()


def analyze(
    source: FrameSource,
    detector: ContactDetector,
    settings: Project,
    frame_interval: float,
    on_progress: Callable[[float], None] | None = None,
    audio_notes: tuple[AudioNote, ...] | None = None,
    motion_tracker=None,
) -> AnalysisResult:
    """Read frames, stabilize contacts, then fuse or quantize into a fresh Project.

    frame_interval is in seconds and controls dropout tolerance and final flushing.
    on_progress receives video seconds after each successfully processed frame.
    Exceptions propagate so callers can report failures without saving partial work.
    """
    finite_number(frame_interval, "Frame interval", 0.001)
    tracker = ContactTracker(release_seconds=max(0.12, frame_interval * 1.5))
    count = with_contacts = 0
    last_seconds = 0.0
    visual_samples = []
    for frame in source.frames():
        frame_calibration = motion_tracker.update(frame.rgb) if motion_tracker else None
        contacts = detector.detect(frame)
        if isinstance(detector, FingertipDetector):
            visual_samples.append(
                VisualSample(frame.seconds, detector.fingertips, frame_calibration)
            )
        tracker.update(frame.seconds, contacts)
        count += 1
        with_contacts += bool(contacts)
        last_seconds = frame.seconds
        if on_progress:
            on_progress(frame.seconds)
    if not count:
        raise ValueError("The video contains no decodable frames.")
    contacts = tracker.finish(last_seconds + frame_interval)
    if audio_notes is not None:
        fused = fuse(audio_notes, contacts, settings, visual_samples=tuple(visual_samples))
        return AnalysisResult(
            replace(settings, notes=fused.notes),
            count,
            with_contacts,
            fused.matched,
            fused.unconfirmed,
            fused.out_of_range,
            tuple(visual_samples),
        )
    notes = quantize(contacts, settings)
    return AnalysisResult(replace(settings, notes=notes), count, with_contacts)
