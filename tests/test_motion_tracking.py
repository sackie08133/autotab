"""Regression coverage for a moving-neck overlay.

The synthetic frame contains textured points inside a calibrated quadrilateral and
then translates the whole image. The tracker should follow that translation while
retaining the same fret-wire bounds. A blank frame verifies the conservative
fallback: the previous box is kept instead of jumping to noise.
"""

import cv2
import numpy as np
import pytest

from autotab.adapters.geometry import FretboardMapper
from autotab.adapters.motion import FretboardMotionTracker
from autotab.domain import Calibration


def _textured_frame(shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Create a repeatable RGB fretboard-like image with enough LK features."""
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(45, 205, 16):
        cv2.line(
            image, (35 + shift_x, y + shift_y), (285 + shift_x, y + shift_y), (80, 170, 110), 2
        )
    for x in range(45, 285, 20):
        cv2.line(
            image, (x + shift_x, 35 + shift_y), (x + shift_x, 210 + shift_y), (190, 150, 90), 2
        )
    return image


def test_motion_tracker_moves_all_four_box_corners_with_neck():
    """A three-pixel translation follows the neck instead of leaving the box behind."""
    initial = Calibration(((0.12, 0.15), (0.88, 0.15), (0.88, 0.85), (0.12, 0.85)))
    mapper = FretboardMapper(initial)
    tracker = FretboardMotionTracker(mapper)
    tracker.update(_textured_frame())
    updated = tracker.update(_textured_frame(6, 3))
    assert updated.corners[0][0] == pytest.approx(initial.corners[0][0] + 6 / 319, abs=0.01)
    assert updated.corners[0][1] == pytest.approx(initial.corners[0][1] + 3 / 239, abs=0.01)
    assert mapper.calibration == updated
    assert updated.first_fret == initial.first_fret
    assert updated.last_fret == initial.last_fret


def test_motion_tracker_keeps_last_good_box_when_flow_fails():
    """A blank frame cannot make a failed tracker replace a valid calibration with noise."""
    initial = Calibration(((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)))
    mapper = FretboardMapper(initial)
    tracker = FretboardMotionTracker(mapper)
    tracker.update(_textured_frame())
    before = tracker.calibration
    after = tracker.update(np.zeros((240, 320, 3), dtype=np.uint8))
    assert after == before
