"""Streamlit session coordination for history, editor resets, and durable recovery.

The editor's seed changes only when explicitly restoring/replacing a score, keeping
Streamlit cell deltas stable on normal reruns. Validated edits enter ProjectHistory;
undo/redo restores the full metadata before widgets instantiate. Each browser session
owns an independent autosave UUID and recovery writes retry after a disk failure.
"""

from uuid import uuid4

import streamlit as st

from autotab.autosave import save_draft
from autotab.domain import Project
from autotab.history import ProjectHistory

DEFAULT_CORNERS = ((0.10, 0.30), (0.90, 0.30), (0.90, 0.70), (0.10, 0.70))


def persist(project: Project) -> None:
    """Save a changed snapshot; surface write failure without losing edits/history."""
    if "draft_id" not in st.session_state:
        st.session_state.draft_id = str(uuid4())
    if st.session_state.get("saved_project") == project:
        return
    try:
        path = save_draft(project, st.session_state.draft_id)
        st.session_state.saved_project = project
        st.session_state.autosave_path = str(path)
        st.session_state.pop("autosave_error", None)
    except OSError as exc:
        st.session_state.autosave_error = str(exc)


def accept_project(project: Project) -> None:
    """Commit valid UI state, then autosave; no-op reruns create no history entries."""
    if "history" not in st.session_state:
        st.session_state.history = ProjectHistory(project)
    else:
        st.session_state.history.commit(project)
    persist(project)


def load_into_session(project: Project, reset_history: bool = True) -> None:
    """Restore a score before widget construction and give its editor a fresh identity.

    Loading an external project starts a new history and autosave identity. Undo/redo
    and applying alternatives retain existing history. Uploaded video is session-only
    and survives undo; users must re-upload it after restarting/recovering a draft.
    """
    if reset_history:
        st.session_state.history = ProjectHistory(project)
        st.session_state.draft_id = str(uuid4())
        st.session_state.pop("saved_project", None)
        st.session_state.pop("analysis_summary", None)
    st.session_state.seed = project
    st.session_state.revision = st.session_state.get("revision", 0) + 1
    for key, value in {
        "title": project.title,
        "bpm": float(project.bpm),
        "offset": float(project.beat_offset_seconds),
        "grid": project.subdivisions,
        "instrument": project.instrument,
        "capo": project.capo,
    }.items():
        st.session_state[key] = value
    calibration = project.calibration
    for i, (x, y) in enumerate(calibration.corners if calibration else DEFAULT_CORNERS):
        st.session_state[f"corner_{i}_x"] = float(x)
        st.session_state[f"corner_{i}_y"] = float(y)
    st.session_state.first_fret = calibration.first_fret if calibration else 0
    st.session_state.last_fret = calibration.last_fret if calibration else 12
    persist(project)


def move_history(direction: str) -> None:
    """Widget callback: move the cursor before the next page builds its widgets."""
    history = st.session_state.history
    project = history.undo() if direction == "undo" else history.redo()
    load_into_session(project, reset_history=False)


def queue_edit(project: Project) -> None:
    """Commit a review action and queue the widget reset for the next clean rerun."""
    accept_project(project)
    st.session_state.pending_project = project
    st.session_state.pending_reset_history = False
    st.rerun()
