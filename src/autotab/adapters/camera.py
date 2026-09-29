"""Validate browser camera recordings and normalize them for existing media adapters.

Browsers choose different MediaRecorder containers, and live WebM recordings often
lack usable duration metadata. This boundary decodes a bounded base64 payload and
transcodes it to a seekable H.264/AAC MP4 before OpenCV sees it. Camera clips then have
the same in-memory file contract as uploaded videos. No device is opened by Python:
camera/microphone permission and recording controls belong to the user's browser.
"""

import base64
import binascii
import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg

MAX_CAMERA_BYTES = 24 * 1024 * 1024
MAX_CAMERA_SECONDS = 120
CAMERA_TYPES = {"video/webm": ".webm", "video/mp4": ".mp4"}


class CameraRecording(BytesIO):
    """A session-local MP4 exposing the same attributes used by uploaded recordings."""

    def __init__(self, content: bytes, recording_id: str):
        """Store bytes, a display filename, and a stable identity for calibration/replay."""
        super().__init__(content)
        self.name = "camera-recording.mp4"
        self.file_id = recording_id
        self.type = "video/mp4"


def recording_from_payload(payload: dict) -> CameraRecording:
    """Validate a completed recorder message and convert its bounded clip to MP4.

    The recording ID is opaque and never becomes a filesystem path. Both size and
    media type are checked before spawning FFmpeg. An invalid recording leaves any
    previous usable clip untouched in the calling UI session.
    """
    if not isinstance(payload, dict):
        raise ValueError("Invalid camera recording message.")
    recording_id, encoded, mime = (payload.get(k) for k in ("id", "data", "mime"))
    if not isinstance(recording_id, str) or not recording_id or len(recording_id) > 100:
        raise ValueError("Invalid camera recording identifier.")
    if not isinstance(mime, str) or mime.split(";")[0] not in CAMERA_TYPES:
        raise ValueError("Camera must record MP4 or WebM video.")
    if (
        not isinstance(encoded, str)
        or not encoded
        or len(encoded) > (MAX_CAMERA_BYTES + 2) // 3 * 4
    ):
        raise ValueError("Camera recording is empty or exceeds the 24 MB limit.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Camera recording data is incomplete. Please record again.") from exc
    if not content or len(content) > MAX_CAMERA_BYTES:
        raise ValueError("Camera recording is empty or exceeds the 24 MB limit.")
    with TemporaryDirectory(prefix="autotab-camera-") as folder:
        source = Path(folder) / ("source" + CAMERA_TYPES[mime.split(";")[0]])
        target = Path(folder) / "recording.mp4"
        source.write_bytes(content)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(source),
            "-t",
            str(MAX_CAMERA_SECONDS),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(target),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"Could not prepare camera recording: {exc}") from exc
        if result.returncode:
            raise ValueError(
                "Could not decode the camera clip. Try recording again or uploading an MP4."
            )
        return CameraRecording(target.read_bytes(), recording_id)
