"""Streamlit presentation and session lifecycle for the AutoTab prototype.

The page walks through user-owned tempo, local video upload, manual fretboard
calibration, audio/visual analysis, and editable/exportable tablature. Expensive
analysis only runs after an explicit button press. Widget reruns never silently
overwrite a draft: a stable table seed is replaced only by loading a project or
creating a new analysis. Valid snapshots autosave locally; downloadable JSON remains
the portable project format. Review and playback panels live in separate UI modules.

Temporary videos are scoped to a context manager and removed after each read or
analysis; only the uploaded bytes remain in Streamlit's session. Processing is
local. The app does not download model weights or upload recordings to a service.
"""

from base64 import b64encode
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from autotab.autosave import available_drafts
from autotab.domain import INSTRUMENTS, Calibration, Note, Project
from autotab.export import to_ascii, to_csv
from autotab.history import ProjectHistory
from autotab.score import to_svg
from autotab.storage import dumps, loads
from autotab.ui.calibration import calibration_controls
from autotab.ui.editor import note_rows, parse_rows
from autotab.ui.media import local_video
from autotab.ui.recording_input import recording_input
from autotab.ui.review import playback_panel, review_panel
from autotab.ui.session import accept_project, load_into_session, move_history


def project_controls() -> Project | None:
    """Render project loading and required tempo; return validated current settings.

    BPM begins empty so the user supplies it explicitly. Changing tempo preserves
    edited beat positions; re-analyzing re-quantizes source timings at the new BPM.
    Loading the bundled demo is the only operation that supplies an example tempo.
    """
    with st.sidebar:
        st.header("Project")
        saved = st.file_uploader(
            "Open saved tab project (.json)",
            type=["json"],
            help="To add an MP4, use Upload video in the main page.",
        )
        if st.button("Load project", disabled=saved is None):
            try:
                load_into_session(loads(saved.getvalue()))
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Try an editable example"):
            load_into_session(
                Project(
                    "Guitar example",
                    100,
                    (
                        Note(6, 0, 0, 1),
                        Note(6, 3, 1, 1),
                        Note(5, 0, 2, 1),
                        Note(5, 2, 3, 1),
                    ),
                )
            )
        drafts = available_drafts()
        if drafts:
            with st.expander("Recover an autosaved draft"):
                selected = st.selectbox(
                    "Saved session",
                    range(len(drafts)),
                    format_func=lambda i: f"{drafts[i][1].title} · {drafts[i][0].stem[:8]}",
                )
                if st.button("Recover draft"):
                    load_into_session(drafts[selected][1])
        st.text_input("Title", key="title", value="Untitled transcription")
        st.selectbox(
            "Instrument",
            list(INSTRUMENTS),
            key="instrument",
            format_func=lambda key: INSTRUMENTS[key].name,
        )
        st.number_input("Capo fret (0 = none)", min_value=0, max_value=12, value=0, key="capo")
        st.number_input(
            "BPM — enter your tempo",
            min_value=1.0,
            max_value=400.0,
            value=None,
            key="bpm",
            step=1.0,
        )
        st.number_input(
            "First beat in video (seconds)", min_value=0.0, value=0.0, step=0.1, key="offset"
        )
        st.selectbox("Grid divisions per beat", list(range(1, 17)), index=3, key="grid")
        st.caption(
            "4 divisions = sixteenth notes when one beat is a quarter note. "
            "Tempo edits keep existing beat positions; rerun analysis to retime the video."
        )
    if st.session_state.bpm is None:
        st.info(
            "Enter BPM in the sidebar to analyze. You can upload or record your video below now."
        )
        return None
    try:
        return Project(
            st.session_state.title,
            st.session_state.bpm,
            beat_offset_seconds=st.session_state.offset,
            subdivisions=st.session_state.grid,
            instrument=st.session_state.instrument,
            capo=st.session_state.capo,
        )
    except ValueError as exc:
        st.error(str(exc))
        return None


def analysis_controls(settings: Project, upload) -> Calibration | None:
    """Render upload/calibration, then execute the local adapters only on request.

    Audio + video is the default path. Silent video requires an explicit mode
    selection. Results are queued for the next rerun so metadata widgets are never
    mutated after Streamlit has instantiated them in the current script run.
    """
    if upload is None:
        history = st.session_state.get("history")
        return history.current.calibration if history else None
    calibration = calibration_controls(upload, settings)
    mode = st.radio("Analysis mode", ["Audio + video", "Vision only"], horizontal=True)
    with st.expander("Analysis settings"):
        model = st.text_input("Hand model file", "models/hand_landmarker.task")
        sample_fps = st.slider("Video analysis frames per second", 5, 30, 15)
        max_seconds = st.number_input("Analyze first N seconds", 1, 300, 120)
        st.caption("Install weights once with: autotab download-model")
    replace_draft = st.checkbox("Replace my current draft with this analysis")
    if st.button(
        "Analyze recording", type="primary", disabled=calibration is None or not replace_draft
    ):
        try:
            model_path = Path(model)
            if not model_path.is_file():
                from autotab.cli import download_model

                with st.spinner("Downloading the hand model for this deployment…"):
                    download_model(model_path)
            from autotab.recording import transcribe_recording

            with (
                local_video(upload) as path,
                st.status("Listening and watching…", expanded=True) as status,
            ):
                progress = st.progress(0, text="Tracking the fretting hand…")
                result = transcribe_recording(
                    path,
                    replace(settings, calibration=calibration),
                    model_path,
                    use_audio=mode == "Audio + video",
                    sample_fps=sample_fps,
                    max_seconds=max_seconds,
                    on_stage=st.write,
                    on_progress=progress.progress,
                )
                status.update(label="Draft ready", state="complete")
            st.session_state.pending_project = result.project
            st.session_state.visual_samples = result.visual_samples
            st.session_state.visual_recording_id = upload.file_id
            st.session_state.visual_calibration = calibration
            st.session_state.pending_reset_history = False
            accept_project(result.project)
            st.session_state.analysis_summary = (
                (
                    f"{len(result.project.notes)} notes · "
                    f"{result.matched_notes} audio/visual matches · "
                    f"{result.unconfirmed_notes} audio-only positions to review · "
                    f"{result.frames_with_contacts}/{result.frames_processed} "
                    "frames with visual contacts."
                )
                if mode == "Audio + video"
                else (
                    f"{len(result.project.notes)} visual guesses. "
                    "Picking and repeated notes were not analyzed."
                )
            )
            if result.out_of_range_notes:
                st.session_state.analysis_summary += (
                    f" {result.out_of_range_notes} out-of-range pitches skipped."
                )
            st.rerun()
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            st.error(f"Analysis could not finish: {exc}")
    return calibration


def render_editor(settings: Project, calibration: Calibration | None) -> None:
    """Render a persistent editable score and export only the current validated rows.

    Original rows are kept stable across reruns because Streamlit stores cell deltas
    against the input table. Users can add rows at the bottom or select rows to delete.
    JSON preserves durations, tuning choice, timing settings, and calibration.
    """
    st.subheader("2. Review and edit your tab")
    if st.session_state.get("analysis_summary"):
        st.info(st.session_state.analysis_summary)
    st.caption(
        "The tab below uses these same string/fret entries. "
        "The highest-pitched string is always the top row."
    )
    st.caption(
        "String 1 is the highest-pitched string. Beat 0 is your first beat. "
        "Frets are relative to the capo. Add rows at the bottom; select rows to delete them. "
        "Durations are in beats."
    )
    seed = st.session_state.get("seed", settings)
    if "history" not in st.session_state:
        st.session_state.history = ProjectHistory(seed)
    history_area = st.container()
    table = pd.DataFrame(
        note_rows(seed.notes), columns=["String", "Fret", "Beat", "Duration", "Source"]
    )
    table = table.astype(
        {
            "String": "Int64",
            "Fret": "Int64",
            "Beat": "float64",
            "Duration": "float64",
            "Source": "str",
        }
    )
    edited = st.data_editor(
        table,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"notes_{st.session_state.get('revision', 0)}",
        disabled=["Source"],
        column_config={
            "String": st.column_config.NumberColumn(
                min_value=1,
                max_value=len(INSTRUMENTS[settings.instrument].open_midi),
                step=1,
                required=True,
            ),
            "Fret": st.column_config.NumberColumn(
                min_value=0, max_value=24 - settings.capo, step=1, required=True
            ),
            "Beat": st.column_config.NumberColumn(
                min_value=0.0, step=1 / settings.subdivisions, required=True
            ),
            "Duration": st.column_config.NumberColumn(
                min_value=0.001, step=1 / settings.subdivisions, required=True
            ),
        },
    )
    try:
        records = edited.astype(object).where(pd.notna(edited), None).to_dict("records")
        project = replace(settings, notes=parse_rows(records, seed.notes), calibration=calibration)
    except ValueError as exc:
        history_controls(history_area)
        st.error(str(exc))
        return
    accept_project(project)
    history_controls(history_area)
    if st.session_state.get("autosave_error"):
        st.warning(
            "Autosave failed; edits remain in this session. Download JSON to keep them. "
            + st.session_state.autosave_error
        )
    else:
        st.caption("Autosaved locally. Use Recover an autosaved draft after reopening the app.")
    st.caption(
        "Source: matched = audio pitch agrees with a visible position; audio = position needs "
        "review; vision = finger-position guess. Your changes export as manual."
    )
    try:
        tab = to_ascii(project)
        score = to_svg(project)
        # Streamlit sanitizes inline SVG out of st.html. Embedding the self-contained
        # vector as an image preserves it and keeps dense bars scrollable at full size.
        encoded_score = b64encode(score.encode("utf-8")).decode("ascii")
        st.html(
            '<div style="overflow-x:auto;max-width:100%">'
            '<img alt="Tablature with numbered bars and rhythm notation" '
            'style="max-width:none;display:block" '
            f'src="data:image/svg+xml;base64,{encoded_score}"></div>'
        )
        st.caption(
            "4/4: each numbered bar contains four quarter-note beats. "
            "Rhythm symbols below the strings use the Duration column: "
            "1 = quarter, 2 = half, 4 = whole, 0.5 = eighth, 0.25 = sixteenth. "
            "Edit Duration to correct a note's length."
        )
        st.download_button("Download graphical tab", score, "autotab.svg", "image/svg+xml")
        with st.expander("Plain-text tab"):
            st.code(tab, language=None)
    except ValueError as exc:
        tab = None
        st.warning(str(exc))
    json_column, csv_column, text_column = st.columns(3)
    json_column.download_button(
        "Save editable project", dumps(project), "autotab-project.json", "application/json"
    )
    csv_column.download_button("Export note CSV", to_csv(project), "autotab-notes.csv", "text/csv")
    if tab is not None:
        text_column.download_button("Export text tab", tab, "autotab.txt", "text/plain")
    st.caption(
        "Download JSON for a portable copy. Autosaves do not include your recording. "
        "Text tabs show note starts; JSON and CSV retain exact durations."
    )
    upload = st.session_state.get("active_recording")
    playback_panel(project, upload)
    review_panel(project, upload)


def history_controls(container) -> None:
    """Render history actions after accepting edits, so enabled states are current.

    The reserved container keeps controls above the table even though they render
    later. Invalid rows still leave history actions available to restore good data.
    """
    with container:
        history = st.session_state.history
        undo, redo, status = st.columns([1, 1, 4])
        undo.button("Undo", disabled=not history.can_undo, on_click=move_history, args=("undo",))
        redo.button("Redo", disabled=not history.can_redo, on_click=move_history, args=("redo",))
        status.caption("Valid edits autosave locally. Undo/redo lasts for this browser session.")


def main() -> None:
    """Build one page rerun, applying queued project replacements before any widgets."""
    st.set_page_config(page_title="AutoTab · Guitar & bass", page_icon="🎸", layout="wide")
    if "pending_project" in st.session_state:
        load_into_session(
            st.session_state.pop("pending_project"),
            reset_history=st.session_state.pop("pending_reset_history", True),
        )
    st.title("AutoTab")
    
    st.caption(
        "Tabs using videos. Guitar (6 String) and Bass (4 String) supported · "
        "Local audio + video analysis"
    )
    st.warning(
        "Keep the fretboard as still as possible during recording. Bends, Slides, Mutes, "
        "Tapping, or chords must be manually reviewed and edited"
    )
    settings = project_controls()
    upload = recording_input()
    if settings is not None:
        calibration = analysis_controls(settings, upload)
        render_editor(settings, calibration)


if __name__ == "__main__":
    main()
