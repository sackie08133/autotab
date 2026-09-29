"""Map fingertip image coordinates onto a manually calibrated fretboard plane.

Four corners define a projective transform into a unit rectangle. Across the
rectangle, strings are evenly spaced; along it, fret boundaries follow the
equal-tempered scale formula rather than equal pixel widths. The same geometry
works for four-string bass and six-string guitar, including a capo because the
normalized relative fret spacing is unchanged after shifting the origin.

This model assumes a stationary, approximately planar neck and straight strings.
It estimates proximity, not physical pressure. UI previews use the inverse
transform so users see precisely the grid used by the detector.
"""

from bisect import bisect_left
from math import floor

import cv2
import numpy as np

from autotab.domain import Calibration, Contact, Fingertip


class FretboardMapper:
    """Reusable forward/inverse projection and string/fret lookup for one calibration."""

    def __init__(self, calibration: Calibration, string_count: int = 6):
        """Build homographies and wire positions once for all frames in a recording.

        Corner coordinates and projected positions are normalized, so video/display
        dimensions do not affect the resulting musical contact.
        """
        if string_count not in (4, 6):
            raise ValueError("This mapper only supports four or six strings.")
        self.calibration = calibration
        self.string_count = string_count
        self.matrix = cv2.getPerspectiveTransform(
            np.asarray(calibration.corners, dtype=np.float32),
            np.asarray([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32),
        )
        self.inverse = np.linalg.inv(self.matrix)
        first = self._wire(calibration.first_fret)
        span = self._wire(calibration.last_fret) - first
        self.boundaries = tuple(
            (self._wire(fret) - first) / span
            for fret in range(calibration.first_fret, calibration.last_fret + 1)
        )

    @staticmethod
    def _wire(fret: int) -> float:
        """Return a fret wire's distance from the origin as a fraction of scale length.

        Each twelve frets halves the remaining vibrating length. Offsetting both
        calibration boundaries handles crops that do not show the nut or capo.
        """
        return 1 - 2 ** (-fret / 12)

    @staticmethod
    def _transform(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
        """Apply homogeneous projection, returning infinity at a vanishing point."""
        point = matrix @ np.array([x, y, 1.0])
        if abs(point[2]) < 1e-10:
            return float("inf"), float("inf")
        return float(point[0] / point[2]), float(point[1] / point[2])

    def contact(self, x: float, y: float) -> Contact | None:
        """Return the nearby string and the next fret wire toward the bridge.

        A finger in the space before wire N sounds fret N. A narrow string-distance
        tolerance rejects ambiguous points between strings; points outside the
        calibrated wire range cannot produce a contact.
        """
        along, across = self._transform(self.matrix, x, y)
        spacing = 1 / (self.string_count - 1)
        tolerance = spacing * 0.325
        if not (0 < along <= 1 and -tolerance <= across <= 1 + tolerance):
            return None
        nearest = int(np.clip(np.floor(across / spacing + 0.5), 0, self.string_count - 1))
        # Reject points between strings rather than assigning every finger to a string.
        if abs(across - nearest * spacing) > tolerance:
            return None
        fret = self.calibration.first_fret + bisect_left(self.boundaries, along)
        return Contact(nearest + 1, fret)

    def update_calibration(self, calibration: Calibration) -> None:
        """Replace the image transform while retaining the same fret/string model."""
        if (
            calibration.first_fret != self.calibration.first_fret
            or calibration.last_fret != self.calibration.last_fret
        ):
            raise ValueError("Tracked calibration cannot change fret-wire bounds.")
        self.__init__(calibration, self.string_count)

    def fingertip(self, x: float, y: float) -> Fingertip | None:
        """Keep continuous near-neck evidence even when a fingertip is between strings.

        Audio fusion can compare the point to each playable same-pitch location.
        This is separate from contact(), whose conservative threshold remains useful
        for silent-video drafts. Completely off-neck points are still rejected.
        """
        along, across = self._transform(self.matrix, x, y)
        string_position = 1 + across * (self.string_count - 1)
        if not (0 <= along <= 1 and 0.25 <= string_position <= self.string_count + 0.75):
            return None
        cell = min(len(self.boundaries) - 1, max(1, bisect_left(self.boundaries, along)))
        left, right = self.boundaries[cell - 1 : cell + 1]
        fret_position = self.calibration.first_fret + cell - 1 + (along - left) / (right - left)
        return Fingertip(string_position, fret_position)

    def image_point(self, along: float, across: float) -> tuple[float, float]:
        """Project a unit-fretboard coordinate back into normalized image coordinates."""
        return self._transform(self.inverse, along, across)

    def overlay_fingertips(self, rgb: np.ndarray, tips: tuple[Fingertip, ...]) -> np.ndarray:
        """Draw retained landmark evidence separately from the tab's proposed contact.

        Yellow dots are projected model fingertips and can fall between strings.
        Labels report their continuous coordinates, helping users spot calibration
        offsets instead of mistaking a pitch-compatible tab choice for a detection.
        """
        image = self.overlay(rgb)
        for tip in tips:
            relative = tip.fret_position - self.calibration.first_fret
            cell = min(len(self.boundaries) - 2, max(0, floor(relative)))
            fraction = relative - cell
            along = self.boundaries[cell] + fraction * (
                self.boundaries[cell + 1] - self.boundaries[cell]
            )
            x, y = self.image_point(along, (tip.string_position - 1) / (self.string_count - 1))
            pixel = (round(x * (image.shape[1] - 1)), round(y * (image.shape[0] - 1)))
            cv2.circle(image, pixel, 5, (255, 230, 50), -1)
            cv2.putText(
                image,
                f"s{tip.string_position:.1f} f{tip.fret_position:.1f}",
                (pixel[0] + 6, pixel[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 230, 50),
                1,
                cv2.LINE_AA,
            )
        return image

    def overlay(self, rgb: np.ndarray) -> np.ndarray:
        """Return a copy annotated with the exact string/wire grid used for inference.

        The caller's RGB image is never modified. OpenCV drawing colors are provided
        in RGB order here because this array is displayed directly as RGB by the UI.
        """
        image = rgb.copy()
        height, width = image.shape[:2]

        def point(x: float, y: float) -> tuple[int, int]:
            """Convert a projected normalized point into a drawable image pixel."""
            px, py = self.image_point(x, y)
            return round(px * (width - 1)), round(py * (height - 1))

        for string in range(self.string_count):
            across = string / (self.string_count - 1)
            cv2.line(image, point(0, across), point(1, across), (78, 220, 160), 1)
        for i, boundary in enumerate(self.boundaries):
            cv2.line(image, point(boundary, 0), point(boundary, 1), (255, 200, 80), 1)
            cv2.putText(
                image,
                str(self.calibration.first_fret + i),
                point(boundary, 0.5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 240, 200),
                1,
                cv2.LINE_AA,
            )
        for i, (x, y) in enumerate(self.calibration.corners):
            pixel = (round(x * (width - 1)), round(y * (height - 1)))
            cv2.circle(image, pixel, 5, (255, 110, 90), -1)
            cv2.putText(
                image,
                str(i + 1),
                (pixel[0] + 6, pixel[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 110, 90),
                2,
            )
        return image
