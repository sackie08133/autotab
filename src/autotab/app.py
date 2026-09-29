"""Package-level Streamlit entrypoint for repositories that expose ``autotab/``.

The preferred Community Cloud entrypoint is the project-root ``app.py``. This
small wrapper supports a repository layout where the package directory itself is
the repository root, so Cloud can use ``autotab/app.py`` after the project is
copied or exported in that form.
"""

from autotab.ui.app import main

main()
