"""Visible video intake independent of the user's BPM entry or saved-project controls.

File upload and direct browser camera recording converge on one active recording.
Switching the source does not destroy the other source's clip or a user's score.
The camera component requests access only after a deliberate button click and sends
a completed bounded recording. Python normalizes it once per capture ID, retaining
the resulting MP4 through normal Streamlit reruns for calibration, analysis, and replay.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from autotab.adapters.camera import recording_from_payload

_recorder = components.declare_component(
    "autotab_camera_recorder",
    path=str(Path(__file__).parent / "camera_component"),
)


def recording_input():
    """Return the selected upload/camera clip while keeping both inputs easy to find.

    This renders before the BPM guard. Camera conversion failures display an error
    without replacing the previous capture, and no media enters project autosaves.
    The active object is shared with downstream review panels via session state.
    """
    st.subheader("1. Add a video")
    st.write(
        "Upload an MP4 recorded on your phone or camera, or record here using your camera "
        "and microphone. Include the fretting hand and the instrument's sound."
    )
    source = st.radio(
        "Video source", ["Upload video", "Record with camera"], horizontal=True, key="video_source"
    )
    # Keep the uploader mounted in both modes so switching to camera does not erase
    # its Streamlit widget state or force a user to upload the same file again.
    with st.expander("Choose an existing video file", expanded=source == "Upload video"):
        upload = st.file_uploader(
            "Upload a guitar or bass video (.mp4)",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="recording_upload",
            help="Choose a video file, not a saved JSON project. Maximum 200 MB.",
        )
    if source == "Upload video":
        active = upload
    else:
        st.caption(
            "Enable camera + microphone, allow access in your browser, then Start recording "
            "and Stop recording. Clips are limited to 2 minutes / 24 MB. "
            "Camera capture works on localhost or HTTPS."
        )
        payload = _recorder(key="camera_recorder", default=None)
        if payload and payload.get("id") != st.session_state.get("camera_payload_id"):
            st.session_state.camera_payload_id = payload.get("id")
            try:
                with st.spinner("Preparing your camera recording…"):
                    st.session_state.camera_recording = recording_from_payload(payload)
                st.session_state.pop("camera_error", None)
            except (ValueError, OSError) as exc:
                st.session_state.camera_error = str(exc)
        if st.session_state.get("camera_error"):
            st.error(st.session_state.camera_error)
        active = st.session_state.get("camera_recording")
        if active is not None:
            st.download_button(
                "Download camera recording as MP4",
                active.getvalue(),
                "camera-recording.mp4",
                "video/mp4",
            )
    st.session_state.active_recording = active
    if active is not None:
        st.video(active.getvalue(), format=getattr(active, "type", "video/mp4"))
        st.caption(
            "Next: enter BPM, align the fretboard grid, then Analyze recording. "
            "Calibration uses this saved video's orientation, including mirrored footage."
        )
    return active
