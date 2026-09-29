"""Headless Streamlit smoke tests for the real page and project session lifecycle.

These tests need no browser, video, or downloaded model. They verify required
BPM, example loading, validated table rendering, exports, and metadata reruns.
Detailed cell conversion is exercised separately in test_editor_export.py.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from autotab.autosave import available_drafts

APP = Path(__file__).parents[1] / "app.py"


def test_app_requires_user_bpm():
    """The initial page asks for tempo without guessing or invoking any ML adapter."""
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.number_input(key="bpm").value is None
    assert any("Enter BPM" in item.value for item in app.info)
    assert any("Tabs using videos" in item.value for item in app.caption)
    assert any("Keep the fretboard as still as possible" in item.value for item in app.warning)


def test_mp4_upload_is_visible_before_bpm_entry():
    """Adding video must not be hidden behind a tempo field or confused with JSON import."""
    app = AppTest.from_file(str(APP)).run(timeout=20)
    labels = [element.proto.label for element in app.get("file_uploader")]
    assert "Upload a guitar or bass video (.mp4)" in labels
    assert "Open saved tab project (.json)" in labels
    uploader = next(e for e in app.get("file_uploader") if ".mp4" in e.proto.label)
    assert ".mp4" in uploader.proto.type
    assert app.number_input(key="bpm").value is None


def test_camera_controls_render_without_bpm_and_keep_file_input():
    """Camera intake is available independently of analysis settings and retains upload state."""
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.radio(key="video_source").set_value("Record with camera").run()
    assert not app.exception
    assert len(app.get("component_instance")) == 1
    assert any("microphone" in caption.value for caption in app.caption)
    assert len(app.get("file_uploader")) == 2


def test_example_loads_and_tempo_change_preserves_notes():
    """A project survives widget reruns and regenerates tab/export metadata correctly."""
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.button[1].click().run()
    assert not app.exception
    assert "Guitar example" in app.code[0].value
    app.number_input(key="bpm").set_value(140).run()
    assert not app.exception
    assert "140 BPM" in app.code[0].value
    assert len(app.dataframe[0].value) == 4


def test_empty_bass_editor_accepts_explicit_tempo_and_capo():
    """Bass and capo settings can be selected without pre-existing notes or weights."""
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.number_input(key="bpm").set_value(90)
    app.number_input(key="capo").set_value(2)
    app.selectbox(key="instrument").select("bass").run()
    assert not app.exception
    assert "Capo 2" in app.code[0].value
    assert "G|" in app.code[0].value
    assert "B|" not in app.code[0].value


def button(app, label):
    """Find a control by its user-facing label rather than sidebar insertion order."""
    return next(control for control in app.button if control.label == label)


def test_tempo_undo_redo_and_autosave_recovery():
    """An edit is immediately undoable, redoable, and recoverable in a new browser session."""
    app = AppTest.from_file(str(APP)).run(timeout=20)
    button(app, "Try an editable example").click().run()
    app.number_input(key="bpm").set_value(140).run()
    assert not button(app, "Undo").disabled
    button(app, "Undo").click().run()
    assert not app.exception
    assert app.number_input(key="bpm").value == 100
    assert not button(app, "Redo").disabled
    button(app, "Redo").click().run()
    assert app.number_input(key="bpm").value == 140
    assert available_drafts()[0][1].bpm == 140
    reopened = AppTest.from_file(str(APP)).run(timeout=20)
    button(reopened, "Recover draft").click().run()
    assert not reopened.exception
    assert reopened.number_input(key="bpm").value == 140
    assert len(reopened.dataframe[0].value) == 4


def test_loop_validation_does_not_break_editor():
    """An invalid playback range produces a useful warning while the editable score survives."""
    app = AppTest.from_file(str(APP)).run(timeout=20)
    button(app, "Try an editable example").click().run()
    app.radio(key="playback_range").set_value("Selected section").run()
    app.number_input(key="loop_end").set_value(1)
    app.number_input(key="loop_start").set_value(2).run()
    assert not app.exception
    assert any("playback range" in warning.value for warning in app.warning)
    assert len(app.dataframe[0].value) == 4


def test_synth_plays_current_edits_and_skips_leading_setup_time(monkeypatch):
    """The visible player must use edited notes and tempo, with a fresh WAV on rerun."""
    from autotab.domain import Note, Project
    from autotab.playback import synthesize
    from autotab.ui import review

    renders = []

    def capture_audio(project, start, end):
        """Render real sound while recording which validated score reached the synth."""
        wav = synthesize(project, start, end)
        renders.append((project, start, end, wav))
        return wav

    monkeypatch.setattr(review, "playback_wav", capture_audio)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.session_state["pending_project"] = Project("Mary", 100, (Note(3, 9, 4.75, 1),))
    app.run()
    assert not app.exception
    assert renders[-1][1:3] == (4.75, 5.75)
    original_audio = renders[-1][3]
    revision = app.session_state["revision"]
    app.session_state[f"notes_{revision}"] = {
        "edited_rows": {0: {"Fret": 7}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert renders[-1][0].notes[0].fret == 7
    assert renders[-1][3] != original_audio
    changed_audio = renders[-1][3]
    app.number_input(key="bpm").set_value(120).run()
    assert len(renders[-1][3]) < len(changed_audio)
    button(app, "Play tab").click().run()
    assert not app.exception
    assert app.get("audio")[0].proto.autoplay
    assert any(e.proto.label == "Download synth audio" for e in app.get("download_button"))


def test_score_is_embedded_as_a_vector_image_with_bar_rhythm_not_a_beat_ruler():
    """Streamlit strips inline SVG; the encoded image must survive its HTML boundary."""
    from base64 import b64decode

    app = AppTest.from_file(str(APP)).run(timeout=20)
    button(app, "Try an editable example").click().run()
    assert not app.exception
    body = app.get("html")[0].proto.body
    assert '<img alt="Tablature with numbered bars and rhythm notation"' in body
    encoded = body.split("base64,", 1)[1].split('"', 1)[0]
    svg = b64decode(encoded).decode()
    assert 'data-duration="quarter"' in svg
    assert "4/4" in svg
    assert "Bar 1" in app.code[0].value
    assert "0.25" not in app.code[0].value
