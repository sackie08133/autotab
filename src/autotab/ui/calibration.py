"""Clickable calibration presentation and validation for the static fretboard mapper.

A small bundled component displays an RGB preview and returns four normalized image
points. It has no npm/CDN dependencies and uses Streamlit's component message protocol.
The browser collects a complete set before submitting; Python validates the polygon
and applies it before coordinate widgets render. Manual coordinates remain available
for fine adjustments, accessibility, and recovery from an invalid click sequence.
"""

import base64
from dataclasses import replace
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components

from autotab.adapters.geometry import FretboardMapper
from autotab.adapters.video import inspect_video, preview_frame
from autotab.domain import INSTRUMENTS, Calibration, Project
from autotab.ui.media import local_video
from autotab.ui.session import DEFAULT_CORNERS

_picker = components.declare_component(
    "autotab_fretboard_picker",
    path=str(Path(__file__).parent / "calibration_component"),
)


def clicked_calibration(value: dict, first: int, last: int) -> Calibration:
    """Convert a component submission into a validated calibration, rejecting bad shapes."""
    try:
        return Calibration(tuple(tuple(point) for point in value["corners"]), first, last)
    except (KeyError, TypeError) as exc:
        raise ValueError("Choose four valid points on the fretboard.") from exc


def calibration_controls(upload, settings: Project) -> Calibration | None:
    """Show preview, click collector, coordinate fallback, and the actual projected grid.

    Click submissions carry a unique event ID to avoid reapplying stale values on
    unrelated reruns. Changing the video or restored project creates a new component
    identity, so a previous recording's partial clicks cannot overwrite calibration.
    """
    with st.expander("Calibrate the fretboard", expanded=True):
        st.write(
            "Click four points on the outer strings in this order: near/highest → "
            "far/highest → far/lowest → near/lowest. Near means closer to the nut. "
            "Wire 0 is the nut or capo; wire numbers are relative to the capo."
        )
        near, far = st.columns(2)
        first = near.number_input("Near fret wire", 0, 23, 0, key="first_fret")
        last = far.number_input("Far fret wire", 1, 24, 12, key="last_fret")
        try:
            with local_video(upload) as path:
                info = inspect_video(path)
                seconds = st.slider(
                    "Preview time (seconds)", 0.0, max(0.01, info.duration - 1 / info.fps), 0.0
                )
                rgb = preview_frame(path, seconds)
            # Bound component payload size while preserving normalized coordinates.
            scale = min(1.0, 1000 / rgb.shape[1])
            preview = cv2.resize(rgb, None, fx=scale, fy=scale)
            _, encoded = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            value = _picker(
                image="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
                key=f"calibration_{upload.file_id}_{st.session_state.get('revision', 0)}",
                default=None,
            )
            if value and value.get("event") != st.session_state.get("last_calibration_event"):
                st.session_state.last_calibration_event = value.get("event")
                try:
                    clicked = clicked_calibration(value, first, last)
                    replace(settings, calibration=clicked)
                    for i, (x, y) in enumerate(clicked.corners):
                        st.session_state[f"corner_{i}_x"] = float(x)
                        st.session_state[f"corner_{i}_y"] = float(y)
                except ValueError as exc:
                    st.error(f"Points not applied: {exc} Click Reset points and try again.")
            corners = []
            with st.expander("Fine-adjust coordinates"):
                for i, column in enumerate(st.columns(4)):
                    column.caption(f"Corner {i + 1}")
                    x = column.number_input(
                        "x", 0.0, 1.0, DEFAULT_CORNERS[i][0], step=0.005, key=f"corner_{i}_x"
                    )
                    y = column.number_input(
                        "y", 0.0, 1.0, DEFAULT_CORNERS[i][1], step=0.005, key=f"corner_{i}_y"
                    )
                    corners.append((x, y))
            calibration = Calibration(tuple(corners), first, last)
            replace(settings, calibration=calibration)
            mapper = FretboardMapper(calibration, len(INSTRUMENTS[settings.instrument].open_midi))
            st.image(mapper.overlay(rgb), caption="Applied grid: green strings, gold fret wires.")
            st.caption("Check that the applied grid follows the fretboard across the clip.")
            return calibration
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return None
