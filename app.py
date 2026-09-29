"""Streamlit entry point. Business logic lives in the autotab package.

Streamlit Community Cloud can launch this file from a repository checkout without
performing an editable install first. Add the local ``src`` directory explicitly so
the same entrypoint works from a clean Cloud environment and from the development
virtual environment.
"""

import sys
from pathlib import Path

source_root = Path(__file__).parent / "src"
if source_root.is_dir() and str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from autotab.ui.app import main  # noqa: E402

main()
