"""Exercise actual media codecs and optional model inference with generated fixtures.

FFmpeg creates a small black video containing a known A3 tone. This verifies real
decoding, timestamp handling, RGB conversion, audio estimation, and full-service
composition without distributing a person's recording. The optional model test
runs only when official weights have already been downloaded; tests never fetch
network assets. A black frame must not hallucinate a visible finger contact.
"""

import base64
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

from autotab.adapters.audio import SAMPLE_RATE, analyze_audio, decode_audio
from autotab.adapters.camera import recording_from_payload
from autotab.adapters.clips import extract_clip
from autotab.adapters.video import OpenCVFrameSource, inspect_video, preview_frame
from autotab.domain import Calibration, Project
from autotab.recording import transcribe_recording

MODEL = Path(__file__).parents[1] / "models" / "hand_landmarker.task"


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Encode a two-second MP4 with real H.264 frames and a known 220 Hz soundtrack."""
    path = tmp_path_factory.mktemp("media") / "known-tone.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=30:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:sample_rate=22050:duration=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return path


def test_real_video_decoder_sampling_and_preview(synthetic_video):
    """Sampling is bounded and monotonic, and previews return the expected RGB shape."""
    info = inspect_video(synthetic_video)
    assert info.duration == pytest.approx(2, abs=0.1)
    frames = list(OpenCVFrameSource(synthetic_video, sample_fps=10, max_seconds=1).frames())
    assert 9 <= len(frames) <= 11
    assert all(a.seconds < b.seconds for a, b in zip(frames, frames[1:], strict=False))
    assert frames[-1].seconds < 1
    assert preview_frame(synthetic_video, 0.5).shape == (240, 320, 3)


def test_real_audio_decode_and_pitch(synthetic_video):
    """The embedded soundtrack decodes at the contracted rate and yields MIDI A3."""
    samples = decode_audio(synthetic_video, 1)
    assert len(samples) == SAMPLE_RATE
    notes = analyze_audio(samples)
    assert notes
    assert all(note.midi == 57 for note in notes)


def test_fractional_video_replay_is_not_rounded_to_seconds(synthetic_video, tmp_path):
    """A quarter-second note segment survives real transcoding and decoding."""
    clip = tmp_path / "fractional.mp4"
    clip.write_bytes(extract_clip(synthetic_video, 0.5, 0.75))
    assert inspect_video(clip).duration == pytest.approx(0.25, abs=0.05)
    assert analyze_audio(decode_audio(clip, 1))[0].midi == 57


def test_browser_webm_capture_becomes_seekable_mp4(synthetic_video, tmp_path):
    """A streamed WebM without finalized duration normalizes for OpenCV and audio analysis."""
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(synthetic_video),
            "-c:v",
            "libvpx",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    recording = recording_from_payload(
        {
            "id": "capture-1",
            "mime": "video/webm;codecs=vp8,opus",
            "data": base64.b64encode(result.stdout).decode(),
        }
    )
    assert recording.file_id == "capture-1"
    assert recording.name.endswith(".mp4")
    path = tmp_path / recording.name
    path.write_bytes(recording.getvalue())
    assert inspect_video(path).duration == pytest.approx(2, abs=0.15)
    assert analyze_audio(decode_audio(path, 1))[0].midi == 57


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"id": "clip", "mime": "video/webm", "data": "not base64"},
        {"id": "clip", "mime": "text/html", "data": "YWJj"},
        {"id": "clip", "mime": "video/webm", "data": ""},
    ],
)
def test_invalid_camera_payloads_are_rejected(payload):
    """Malformed or wrong-type component messages fail before decoder execution."""
    with pytest.raises(ValueError):
        recording_from_payload(payload)


def test_missing_audio_reports_error(tmp_path):
    """A silent video produces a clear error instead of silently switching analysis mode."""
    path = tmp_path / "silent.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=0.5",
            "-an",
            "-c:v",
            "libx264",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    with pytest.raises(ValueError, match="No usable audio"):
        decode_audio(path, 1)


@pytest.mark.skipif(not MODEL.is_file(), reason="Run autotab download-model for model smoke test")
def test_full_recording_service_with_real_model(synthetic_video):
    """Actual model inference and audio decoding compose into reviewable audio-only notes."""
    calibration = Calibration(((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)))
    progress = []
    result = transcribe_recording(
        synthetic_video,
        Project("Smoke", 120, calibration=calibration),
        MODEL,
        sample_fps=5,
        max_seconds=1,
        on_progress=progress.append,
    )
    assert result.project.notes
    assert result.frames_with_contacts == 0
    assert all(note.source == "audio" for note in result.project.notes)
    assert result.matched_notes == 0
    assert progress[-1] == 1.0
