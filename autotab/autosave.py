"""Atomic, local recovery files for independent browser-session drafts.

Each session uses a UUID filename, so two tabs do not silently overwrite one another.
Only validated Project data is written; recording bytes and model weights are never
saved. Writes use a temporary sibling, fsync, and atomic replacement to keep the last
good draft readable if writing fails. The UI offers explicit recovery on startup;
opening the page never replaces a prior draft with an empty score.
"""

import os
import tempfile
from pathlib import Path
from uuid import UUID

from autotab.domain import Project
from autotab.storage import dumps, loads


def draft_directory() -> Path:
    """Locate project-local recovery storage, with an override for tests/deployment."""
    return Path(os.environ.get("AUTOTAB_DRAFT_DIR", str(Path.cwd() / ".autotab" / "drafts")))


def save_draft(project: Project, session_id: str, directory: Path | None = None) -> Path:
    """Atomically replace this session's recovery snapshot and return its path.

    UUID validation ensures filenames cannot escape the chosen directory. Errors
    propagate so the UI can report an unsaved state and retain in-memory history.
    """
    name = str(UUID(session_id))
    directory = directory if directory is not None else draft_directory()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(dumps(project))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def available_drafts(directory: Path | None = None) -> list[tuple[Path, Project]]:
    """List readable recovery projects newest first, ignoring incomplete/corrupt files.

    Invalid drafts remain on disk for inspection. A failed file must not prevent
    other recoverable sessions from appearing in the sidebar.
    """
    directory = directory if directory is not None else draft_directory()
    found = []
    for path in directory.glob("*.json"):
        try:
            found.append((path, loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda item: item[0].stat().st_mtime, reverse=True)
