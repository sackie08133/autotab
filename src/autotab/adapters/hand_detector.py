"""Adapt MediaPipe's pretrained Hand Landmarker to the ContactDetector protocol.

The neural model locates hand landmarks; a separate FretboardMapper estimates
which string and fret each non-thumb fingertip is near. These are visual contact
hypotheses, not picked notes. Fusion cross-checks them against acoustic pitch.

Model lifetime is explicit through a context manager. Video mode maintains
tracking state and needs strictly increasing millisecond timestamps. The adapter
handles that unit conversion without exposing MediaPipe objects to the domain.
"""

from pathlib import Path

import mediapipe as mp

from autotab.adapters.geometry import FretboardMapper
from autotab.domain import Contact, Fingertip
from autotab.ports import VideoFrame


class MediaPipeContactDetector:
    """A stateful per-video detector; create a new instance for each analysis run."""

    def __init__(self, model_path: Path, mapper: FretboardMapper):
        """Load an existing model and enable two-hand detection at moderate thresholds.

        Two hands are considered so the picking hand does not prevent detection of
        the fretting hand. No left/right assumption is made about the player.
        """
        if not model_path.is_file():
            raise ValueError("Hand model missing. Run: autotab download-model")
        self.mapper = mapper
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._model = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1
        self._fingertips: tuple[Fingertip, ...] = ()

    @property
    def fingertips(self) -> tuple[Fingertip, ...]:
        """Expose unrounded positions so audio fusion does not lose near-string evidence."""
        return self._fingertips

    def __enter__(self):
        """Expose the initialized adapter inside a resource-managed analysis block."""
        return self

    def __exit__(self, *_):
        """Release native model resources on success or failure without swallowing errors."""
        self._model.close()

    def detect(self, frame: VideoFrame) -> tuple[Contact, ...]:
        """Infer fingertips, map them to the neck, and choose the best-supported hand.

        The thumb is excluded because its role differs and is frequently behind the
        neck. Barre contacts and string pressure need a richer future contact model.
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.rgb)
        timestamp_ms = max(self._last_ms + 1, round(frame.seconds * 1000))
        self._last_ms = timestamp_ms
        result = self._model.detect_for_video(image, timestamp_ms)
        candidates: list[tuple[tuple[Fingertip, ...], set[Contact]]] = []
        for hand in result.hand_landmarks:
            points = tuple(
                point
                for index in (8, 12, 16, 20)
                if (point := self.mapper.fingertip(hand[index].x, hand[index].y)) is not None
            )
            contacts = {
                contact
                for index in (8, 12, 16, 20)
                if (contact := self.mapper.contact(hand[index].x, hand[index].y)) is not None
            }
            candidates.append((points, contacts))
        # The hand with most fingertips on the calibrated neck is likely the fretting hand.
        points, best = max(
            candidates, key=lambda pair: (len(pair[0]), len(pair[1])), default=((), set())
        )
        self._fingertips = points
        return tuple(sorted(best, key=lambda c: (c.string, c.fret)))
