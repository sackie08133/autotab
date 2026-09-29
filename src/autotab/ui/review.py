"""Simple review and playback panels for the current validated score.

Selecting a review-table row seeks a short segment of the original recording and
shows its proposed location on a tracked, calibrated frame. Alternative position buttons
preserve pitch and enter the same undo/autosave flow as table edits. Synthesized tab
playback and source playback share a beat range, but remain independent players so
the user can compare them without promising sample-accurate synchronized playback.
"""

from dataclasses import replace
from io import BytesIO
from math import ceil

import cv2
import pandas as pd
import streamlit as st

from autotab.adapters.clips import extract_clip
from autotab.adapters.geometry import FretboardMapper
from autotab.adapters.video import inspect_video, preview_frame
from autotab.domain import INSTRUMENTS, Project
from autotab.playback import synthesize
from autotab.review import alternatives, change_position, pitch_name, replay_interval, review_reason
from autotab.ui.media import local_video
from autotab.ui.session import queue_edit


@st.cache_data(max_entries=8, show_spinner=False)
def playback_wav(project: Project, start: float, end: float) -> bytes:
    """Cache bounded synthetic audio by the exact immutable score and chosen beat range."""
    return synthesize(project, start, end)


@st.cache_data(max_entries=4, show_spinner="Preparing source segment…")
def source_clip(content: bytes, suffix: str, start: float, end: float) -> bytes:
    """Cache fractional-second replays by recording contents and exact time bounds."""
    upload = BytesIO(content)
    upload.name = "source" + suffix
    with local_video(upload) as path:
        return extract_clip(path, start, end)


def review_panel(project: Project, upload) -> None:
    """Let a user select any note, inspect uncertainty, replay it, and choose an alternative."""
    if not project.notes:
        return
    st.subheader("4. Check notes against the recording")
    reasons = [review_reason(note) for note in project.notes]
    st.caption(
        f"{sum(bool(reason) for reason in reasons)} notes flagged. "
        "Click a row to review or replay it. A match is evidence, not a guarantee."
    )
    rows = [
        {
            "Note": i + 1,
            "Beat": n.beat,
            "String": n.string,
            "Fret": n.fret,
            "Pitch": pitch_name(n, project),
            "Review": "⚠ " + reasons[i]
            if reasons[i]
            else "Matched"
            if n.source == "matched"
            else "Manual / reviewed",
        }
        for i, n in enumerate(project.notes)
    ]
    event = st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"review_{st.session_state.get('revision', 0)}",
    )
    selected = event.selection.rows
    if not selected or selected[0] >= len(project.notes):
        st.caption("Select a note above to see its replay and alternate positions.")
        return
    index = selected[0]
    note = project.notes[index]
    if reasons[index]:
        st.warning(reasons[index])
    options = alternatives(note, project)
    if options:
        choice = st.selectbox(
            "Other positions for the same pitch",
            options,
            format_func=lambda p: f"String {p[0]} · fret {p[1]}",
            key=f"alternative_{index}_{st.session_state.get('revision', 0)}",
        )
        if st.button("Use this position"):
            try:
                queue_edit(change_position(project, index, *choice))
            except ValueError as exc:
                st.error(str(exc))
    if reasons[index] and st.button("Mark this note reviewed"):
        notes = list(project.notes)
        notes[index] = replace(note, source="manual", review_reason="")
        queue_edit(replace(project, notes=tuple(notes)))
    if upload is None:
        st.info(
            "Upload the original recording to replay this note. "
            "Recovered drafts do not store videos."
        )
        return
    try:
        with local_video(upload) as path:
            info = inspect_video(path)
            start, end = replay_interval(note, project, info.duration)
            onset = project.beat_offset_seconds + note.beat * 60 / project.bpm
            rgb = preview_frame(path, min(onset, info.duration - 1 / info.fps))
        st.video(source_clip(upload.getvalue(), ".mp4", start, end), autoplay=True)
        st.caption(f"Source segment {start:.2f}–{end:.2f}s. Press Play again to replay.")
        if project.calibration:
            mapper = FretboardMapper(
                project.calibration, len(INSTRUMENTS[project.instrument].open_midi)
            )
            sample_mapper = mapper
            frame = mapper.overlay(rgb)
            samples = st.session_state.get("visual_samples", ())
            if (
                samples
                and st.session_state.get("visual_recording_id") == upload.file_id
                and st.session_state.get("visual_calibration") == project.calibration
            ):
                closest = min(samples, key=lambda sample: abs(sample.seconds - onset))
                if abs(closest.seconds - onset) <= 0.25:
                    if closest.calibration is not None:
                        sample_mapper = FretboardMapper(
                            closest.calibration, len(INSTRUMENTS[project.instrument].open_midi)
                        )
                    frame = sample_mapper.overlay_fingertips(rgb, closest.fingertips)
                    st.caption(
                        "Yellow dots: model fingertips near this onset. "
                        "Red circle: proposed tab position. These are different evidence."
                    )
            first = project.calibration.first_fret
            if first < note.fret <= project.calibration.last_fret or note.fret == first == 0:
                relative = note.fret - first
                along = (
                    0
                    if note.fret == 0
                    else sum(sample_mapper.boundaries[relative - 1 : relative + 1]) / 2
                )
                x, y = sample_mapper.image_point(
                    along, (note.string - 1) / (sample_mapper.string_count - 1)
                )
                pixel = (round(x * (frame.shape[1] - 1)), round(y * (frame.shape[0] - 1)))
                cv2.circle(frame, pixel, 12, (255, 65, 85), 3)
                st.image(frame, caption="Red circle: proposed tab position at the note onset.")
            else:
                st.info("This note's position is outside the calibrated fret range.")
    except (ValueError, OSError) as exc:
        st.warning(str(exc))


def playback_panel(project: Project, upload) -> None:
    """Play the current edited score, or a selected loop, through the reference synth.

    Whole-tab playback starts at the first note, skipping recording setup silence.
    Every rerun derives audio from the current validated project, so edits to notes,
    tempo, capo, or instrument invalidate cached sound automatically. Autoplay is
    requested only when Play tab is clicked; native controls remain available if
    the browser blocks it. Long scores can be reviewed in bounded sections.
    """
    if not project.notes:
        return
    st.subheader("3. Play your tab")
    st.caption(
        "Hear the current tab with a simple synth at your BPM. "
        "Note edits update the sound automatically."
    )
    length = max(n.beat + n.duration for n in project.notes)
    mode = st.radio(
        "Playback range", ["Whole tab", "Selected section"], horizontal=True, key="playback_range"
    )
    start = min(n.beat for n in project.notes)
    end = length
    if mode == "Selected section":
        left, right = st.columns(2)
        start = left.number_input(
            "Loop start beat",
            min_value=0.0,
            value=float(start),
            step=1 / project.subdivisions,
            key="loop_start",
        )
        end = right.number_input(
            "Loop end beat",
            min_value=0.001,
            value=float(min(ceil(length), start + project.bpm * 2)),
            step=1 / project.subdivisions,
            key="loop_end",
        )
    loop = st.checkbox("Repeat playback", value=False)
    try:
        audio = playback_wav(project, start, end)
        play = st.button("Play tab", type="primary")
        st.audio(audio, format="audio/wav", loop=loop, autoplay=play)
        st.download_button("Download synth audio", audio, "autotab-synth.wav", "audio/wav")
        st.caption(
            "Use the player's pause control to stop. For clips over 120 seconds, select a section."
        )
        if upload is not None:
            with local_video(upload) as path:
                duration = inspect_video(path).duration
            source_start = project.beat_offset_seconds + start * 60 / project.bpm
            source_end = project.beat_offset_seconds + end * 60 / project.bpm
            if source_start < duration:
                if st.button("Prepare recording loop"):
                    st.session_state.source_loop = (
                        upload.file_id,
                        source_start,
                        source_end,
                        source_clip(
                            upload.getvalue(), ".mp4", source_start, min(duration, source_end)
                        ),
                    )
                prepared = st.session_state.get("source_loop")
                if prepared and prepared[:3] == (upload.file_id, source_start, source_end):
                    st.video(prepared[3], loop=loop)
            else:
                st.info("This loop starts after the uploaded recording ends.")
    except (ValueError, OSError) as exc:
        st.warning(str(exc))
