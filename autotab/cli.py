"""Small command-line utilities for setup and launching the local editor.

Commands are deliberately thin: the CLI never duplicates transcription logic.
Model download is explicit, versioned, atomic, and separate from analysis.
Once dependencies and weights are installed, videos are processed locally.
The app binds to loopback so launching does not expose an upload server on LAN.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def download_model(destination: Path) -> None:
    """Fetch Google's version-1 hand model without replacing an existing file.

    Write to a temporary sibling and rename only after the transfer completes.
    A failed download leaves no partial model at the expected destination.
    The weights retain their upstream license; they are not project source code.
    """
    if destination.exists():
        print(f"Model already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if temporary.stat().st_size < 1_000_000:
            raise ValueError("Downloaded model is unexpectedly small; refusing to install it.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Model saved to {destination}")


def main() -> None:
    """Parse a setup/run command and propagate a useful exit status to the shell.

    The run command uses this interpreter's Streamlit module, ensuring that the
    app and its dependencies come from the same virtual environment.
    """
    parser = argparse.ArgumentParser(description="Video + audio assisted guitar and bass tabs")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-model", help="Download pretrained hand weights")
    download.add_argument("--output", type=Path, default=Path("models/hand_landmarker.task"))
    commands.add_parser("run", help="Open the local Streamlit editor")
    args = parser.parse_args()
    if args.command == "download-model":
        try:
            download_model(args.output)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Model download failed: {exc}\n")
    else:
        script = Path(__file__).parent / "ui" / "app.py"
        env = {**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(script),
                    "--server.address=127.0.0.1",
                ],
                env=env,
            )
        )


if __name__ == "__main__":
    main()
