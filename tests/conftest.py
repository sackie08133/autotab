"""Keep UI autosave tests isolated from real user recovery files.

Every test gets its own writable draft directory. Tests can restart Streamlit within
that directory to verify recovery without touching production session snapshots.
"""

import pytest


@pytest.fixture(autouse=True)
def isolated_drafts(tmp_path, monkeypatch):
    """Redirect autosave storage before application code creates any session state."""
    monkeypatch.setenv("AUTOTAB_DRAFT_DIR", str(tmp_path / "drafts"))
