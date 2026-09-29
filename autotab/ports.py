"""Minimal contracts between transcription services and external integrations.

A FrameSource yields timestamped RGB images; a ContactDetector proposes visible
positions for each frame. Protocols use structural typing, so tests can supply
simple fakes and a future model can replace MediaPipe without inheritance.
Images are opaque here: only the actual video/model adapters depend on NumPy.
Resource ownership stays with the caller, normally through context managers.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from autotab.domain import Contact, Fingertip


@dataclass(frozen=True)
class VideoFrame:
    """An RGB image and its absolute presentation timestamp in video seconds."""

    seconds: float
    rgb: Any


class FrameSource(Protocol):
    """A decoder or synthetic fixture yielding frames in strict timestamp order."""

    def frames(self) -> Iterable[VideoFrame]:
        """Yield samples lazily; adapters must release decoder resources after iteration."""
        ...


class ContactDetector(Protocol):
    """A swappable visual model/geometric estimator independent of beat timing."""

    def detect(self, frame: VideoFrame) -> tuple[Contact, ...]:
        """Return proposed fretted contacts, or an empty tuple when no evidence exists."""
        ...


@runtime_checkable
class FingertipDetector(Protocol):
    """Optional richer evidence channel for detectors supporting continuous positions.

    Older/fake contact-only detectors remain valid. The pipeline reads this property
    immediately after detect(), before another frame replaces the model's evidence.
    """

    @property
    def fingertips(self) -> tuple[Fingertip, ...]:
        """Return projected positions from the most recently processed frame."""
        ...
