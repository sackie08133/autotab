"""Small-motion fretboard tracking for a calibrated video.

The user still supplies the first-frame quadrilateral. This adapter follows that
quadrilateral through later frames with sparse Lucas–Kanade optical flow and a
RANSAC homography. It deliberately tracks neck texture rather than the hand:
fret wires and inlays provide stable points while fingers can occlude strings.
Tracking is conservative. If fewer than four reliable points remain, the previous
calibration is retained and a fresh feature search starts on the next frame.
"""

import cv2
import numpy as np

from autotab.adapters.geometry import FretboardMapper
from autotab.domain import Calibration


class FretboardMotionTracker:
    """Track one calibrated neck with bounded, frame-to-frame projective motion."""

    def __init__(self, mapper: FretboardMapper, max_points: int = 160):
        """Initialize an idle tracker; the first frame seeds texture points."""
        self.mapper = mapper
        self.max_points = max_points
        self._previous_gray: np.ndarray | None = None
        self._previous_points: np.ndarray | None = None
        self._calibration = mapper.calibration

    @property
    def calibration(self) -> Calibration:
        """Return the most recent stable calibration used by the mapper."""
        return self._calibration

    def _seed(self, gray: np.ndarray) -> None:
        """Find textured neck points inside the current quadrilateral."""
        height, width = gray.shape[:2]
        polygon = np.asarray(
            [
                [round(x * (width - 1)), round(y * (height - 1))]
                for x, y in self._calibration.corners
            ],
            dtype=np.int32,
        )
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, polygon, 255)
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_points, qualityLevel=0.01, minDistance=8, mask=mask
        )
        corners = polygon.astype(np.float32).reshape(-1, 1, 2)
        self._previous_points = (
            corners if points is None else np.concatenate((points, corners), axis=0)
        )

    def update(self, rgb: np.ndarray) -> Calibration:
        """Follow the box one frame and update the mapper before detection.

        The homography is estimated in pixels then converted back to normalized
        corners. Updates are blended at 70% new / 30% previous for a stable overlay.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if float(gray.std()) < 2.0:
            # There is no texture from which motion can be inferred. Keep the
            # last good box and force a fresh seed when detail returns.
            self._previous_gray = gray
            self._previous_points = None
            return self._calibration
        if self._previous_gray is None or self._previous_points is None:
            self._seed(gray)
            self._previous_gray = gray
            return self._calibration
        current, status, error = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            self._previous_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if current is not None and status is not None:
            valid = status.reshape(-1).astype(bool)
            if error is not None:
                valid &= error.reshape(-1) < 45
            before = self._previous_points.reshape(-1, 2)[valid]
            after = current.reshape(-1, 2)[valid]
        else:
            before = after = np.empty((0, 2), dtype=np.float32)
        # This feature is for slight camera/neck movement. Reject a wholesale
        # optical-flow hallucination (often caused by a blank or blurred frame)
        # before it can move the box across the image.
        plausible_motion = len(before) >= 4 and float(
            np.median(np.linalg.norm(after - before, axis=1))
        ) <= max(rgb.shape[:2]) * 0.15
        if plausible_motion:
            homography, inliers = cv2.findHomography(before, after, cv2.RANSAC, 5.0)
            if homography is not None and inliers is not None and int(inliers.sum()) >= 4:
                self._apply_homography(homography, rgb.shape[1], rgb.shape[0])
                self._previous_points = after.reshape(-1, 1, 2)
            else:
                self._seed(gray)
        else:
            self._seed(gray)
        self._previous_gray = gray
        return self._calibration

    def _apply_homography(self, homography: np.ndarray, width: int, height: int) -> None:
        """Apply and smooth a valid image-space homography to the four corners."""
        old = np.asarray(self._calibration.corners, dtype=np.float32).reshape(-1, 1, 2)
        old[:, :, 0] *= width - 1
        old[:, :, 1] *= height - 1
        moved = cv2.perspectiveTransform(old, homography).reshape(-1, 2)
        normalized = np.clip(moved / np.asarray([width - 1, height - 1], dtype=np.float32), 0, 1)
        blended = 0.3 * np.asarray(self._calibration.corners) + 0.7 * normalized
        try:
            calibration = Calibration(
                tuple(map(tuple, blended)),
                self._calibration.first_fret,
                self._calibration.last_fret,
            )
        except ValueError:
            return
        self._calibration = calibration
        self.mapper.update_calibration(calibration)
