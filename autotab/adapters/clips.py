"""Accurate bounded source-video segments for note replay and musical loops.

Streamlit's built-in media start/end parameters round to integer seconds, which loses
short note boundaries. This adapter therefore transcodes the chosen fractional-second
interval into a small MP4. Native controls then play/loop the entire clip accurately.
FFmpeg runs locally with optional audio mapping, explicit timeout, and temporary-file
cleanup. Rendering is capped to the same two-minute limit as synthesized playback.
"""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg

from autotab.domain import finite_number


def extract_clip(path: Path, start: float, end: float) -> bytes:
    """Return an H.264/AAC MP4 for [start, end) seconds, including silent video support.

    Input seeking plus transcoding provides accurate seeking after the preceding
    keyframe. Resolution is capped at 960 pixels wide to keep repeated reviews light.
    The source recording is never modified, and every subprocess path is an argument.
    """
    finite_number(start, "Clip start")
    finite_number(end, "Clip end")
    if not 0 < end - start <= 120:
        raise ValueError("Choose a video segment between zero and 120 seconds long.")
    with TemporaryDirectory(prefix="autotab-clip-") as directory:
        target = Path(directory) / "clip.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            str(path),
            "-t",
            str(end - start),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale=min(960\\,iw):-2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
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
                timeout=max(30, (end - start) * 3),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"Could not prepare video replay: {exc}") from exc
        if result.returncode:
            raise ValueError(
                "Could not prepare video replay: " + result.stderr.decode(errors="replace")[-300:]
            )
        return target.read_bytes()
