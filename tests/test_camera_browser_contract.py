"""Run recorder lifecycle contract tests without requesting real hardware access.

The app remains Python-based with no frontend build step. When Node is available for
development, its built-in VM/test runner exercises the shipped recorder JavaScript
against fake devices. Platforms without Node skip this extra check; Python media and
UI tests still run. Real camera permissions and capture remain a manual browser test.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Optional Node recorder lifecycle checks")
def test_recorder_javascript_lifecycle():
    """Verify permission gating, clip delivery, track cleanup, and denial recovery."""
    result = subprocess.run(
        [
            shutil.which("node"),
            "--test",
            str(Path(__file__).with_name("camera_component.test.cjs")),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
