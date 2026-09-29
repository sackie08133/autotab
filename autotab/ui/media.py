"""Short-lived local files for UI uploads shared by analysis and review panels.

OpenCV and FFmpeg need a real path; browser uploads are byte buffers. Materializing
them inside a context manager avoids Windows open-file conflicts and ensures that
recordings are not included in autosaves or left in the workspace after processing.
"""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def local_video(upload):
    """Expose a closed temporary file containing an upload for one decoder operation."""
    with TemporaryDirectory(prefix="autotab-") as directory:
        path = Path(directory) / ("video" + Path(upload.name).suffix)
        path.write_bytes(upload.getbuffer())
        yield path
