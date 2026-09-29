"""Local video metadata, previews, and bounded frame sampling through OpenCV.

Decoding stays out of the pipeline: this adapter produces RGB VideoFrame objects
with timestamps in seconds. Presentation timestamps are preferred for variable
frame rate; a monotonic nominal-frame-rate fallback covers codecs without them.
Frames are grabbed sequentially and converted only at the requested sample rate.

Every capture is released in a finally block, including failed reads and early
generator closure. The UI limits analysis duration to keep prototype work bounded.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import cv2

from autotab.domain import finite_number
from autotab.ports import VideoFrame


@dataclass(frozen=True)
class VideoInfo:
    """Decoder metadata: nominal FPS, duration in seconds, and image pixel dimensions."""

    fps: float
    duration: float
    width: int
    height: int


def _open(path: Path):
    """Open a decoder or raise a user-facing format error without leaking its handle."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot decode this video. Try an MP4 with H.264 video.")
    return capture


def inspect_video(path: Path) -> VideoInfo:
    """Read and validate timing metadata without retaining an open capture.

    Unknown or nonsensical frame rates are rejected rather than inventing timing,
    because onset alignment and user-entered BPM depend on meaningful seconds.
    """
    capture = _open(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not isfinite(fps) or fps <= 0 or not isfinite(count) or count <= 0:
            raise ValueError("Video has invalid frame-rate or duration metadata.")
        return VideoInfo(
            fps,
            count / fps,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def preview_frame(path: Path, seconds: float = 0):
    """Seek approximately to a video timestamp and return one RGB calibration image."""
    capture = _open(path)
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        success, bgr = capture.read()
        if not success:
            raise ValueError("Cannot decode the preview frame at this time.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


class OpenCVFrameSource:
    """Lazy decoder implementing FrameSource for one local, duration-limited video."""

    def __init__(self, path: Path, sample_fps: float = 15, max_seconds: float = 120):
        """Validate requested sampling and cap it at the video's nominal frame rate."""
        finite_number(sample_fps, "Sample FPS", 1)
        finite_number(max_seconds, "Analysis duration", 0.1)
        self.path = path
        self.info = inspect_video(path)
        self.sample_fps = min(sample_fps, self.info.fps)
        self.max_seconds = max_seconds

    def frames(self) -> Iterator[VideoFrame]:
        """Yield chronological RGB samples until end of stream or the analysis limit.

        Seeking each sample can skip inconsistently between keyframes, so this method
        advances sequentially. Sampling lowers inference work while preserving the
        recording's time origin for alignment with its decoded audio track.
        """
        capture = _open(self.path)
        index = 0
        next_sample = 0.0
        previous_time = -1.0
        try:
            while capture.grab():
                # Prefer presentation timestamps to support variable-frame-rate video.
                measured = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                seconds = (
                    measured
                    if isfinite(measured) and measured > previous_time
                    else index / self.info.fps
                )
                if seconds <= previous_time:
                    seconds = previous_time + 1 / self.info.fps
                previous_time = seconds
                index += 1
                if seconds >= self.max_seconds:
                    break
                if seconds + 1e-6 < next_sample:
                    continue
                success, bgr = capture.retrieve()
                if not success:
                    raise ValueError(f"Video decoding failed near {seconds:.2f} seconds.")
                next_sample = seconds + 1 / self.sample_fps
                yield VideoFrame(seconds, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
