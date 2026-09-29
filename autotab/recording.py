"""File-based application entry point that composes the optional media adapters.

pipeline.py remains independent of codecs, filesystem paths, and model runtimes.
This module is the composition layer: it creates decoders and a detector, enforces
shared timing limits, and closes native resources before returning a draft. Both
the web UI and a future batch CLI can call the same function.

Heavy imports are inside the entry point, so domain-only consumers can import
AutoTab without installing the app extra. Progress is delivered as stage messages
and fractions rather than UI-specific objects.
"""

from collections.abc import Callable
from pathlib import Path

from autotab.domain import INSTRUMENTS, Project
from autotab.pipeline import AnalysisResult, analyze


def transcribe_recording(
    path: Path,
    settings: Project,
    model_path: Path,
    *,
    use_audio: bool = True,
    sample_fps: float = 15,
    max_seconds: float = 180,
    on_stage: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> AnalysisResult:
    """Analyze one calibrated local recording and return an editable draft.

    The BPM and offset are required through settings; no tempo estimation occurs.
    Audio and video start at the recording origin and share the same duration bound.
    Missing model/audio/frames raise a recoverable error instead of silently changing
    modes. Progress callbacks are optional and must not mutate analysis inputs.
    """
    from autotab.adapters.audio import analyze_audio, decode_audio
    from autotab.adapters.geometry import FretboardMapper
    from autotab.adapters.hand_detector import MediaPipeContactDetector
    from autotab.adapters.motion import FretboardMotionTracker
    from autotab.adapters.video import OpenCVFrameSource

    if settings.calibration is None:
        raise ValueError("Calibrate the fretboard before analyzing a recording.")
    if not model_path.is_file():
        raise ValueError("Hand model missing. Run: autotab download-model")

    source = OpenCVFrameSource(path, sample_fps, max_seconds)
    duration = min(source.info.duration, max_seconds)
    if settings.beat_offset_seconds >= duration:
        raise ValueError("First beat time must fall inside the analyzed part of the video.")
    audio_notes = None
    if use_audio:
        if on_stage:
            on_stage("Finding pitches and note attacks in the audio…")
        audio_notes = analyze_audio(
            decode_audio(path, duration),
            settings.instrument,
            settings.capo,
        )
    if on_stage:
        on_stage("Tracking the fretting hand and matching pitches…")
    mapper = FretboardMapper(settings.calibration, len(INSTRUMENTS[settings.instrument].open_midi))
    motion_tracker = FretboardMotionTracker(mapper)
    with MediaPipeContactDetector(model_path, mapper) as detector:
        result = analyze(
            source,
            detector,
            settings,
            1 / source.sample_fps,
            (lambda seconds: on_progress(min(1.0, seconds / duration))) if on_progress else None,
            audio_notes=audio_notes,
            motion_tracker=motion_tracker,
        )
    if on_progress:
        on_progress(1.0)
    return result
