"""Bounded undo/redo for complete immutable project snapshots.

History includes notes, timing, tuning, and calibration so reversing an edit restores
a coherent score. No UI or disk dependencies live here. A new edit after undo drops
the redo branch; unchanged reruns never add duplicate history entries. Disk recovery
is deliberately separate: only the latest valid snapshot is autosaved, while undo
history lasts for the current browser session.
"""

from autotab.domain import Project


class ProjectHistory:
    """Track up to a bounded number of accepted project states in one session."""

    def __init__(self, project: Project, limit: int = 100):
        """Start with an accepted baseline and a positive snapshot retention limit."""
        if limit < 2:
            raise ValueError("History needs room for at least two states.")
        self._states = [project]
        self._cursor = 0
        self.limit = limit

    @property
    def current(self) -> Project:
        """Return the currently selected immutable snapshot."""
        return self._states[self._cursor]

    @property
    def can_undo(self) -> bool:
        """Whether an older retained state exists."""
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        """Whether an undone state remains on the current branch."""
        return self._cursor < len(self._states) - 1

    def commit(self, project: Project) -> bool:
        """Record a changed valid score, dropping redo states and oldest overflow.

        Returns whether history changed, allowing callers to avoid redundant disk
        writes on ordinary widget reruns and media playback selections.
        """
        if project == self.current:
            return False
        self._states = self._states[: self._cursor + 1] + [project]
        self._states = self._states[-self.limit :]
        self._cursor = len(self._states) - 1
        return True

    def undo(self) -> Project:
        """Move to the preceding snapshot, or keep the earliest one at the boundary."""
        self._cursor = max(0, self._cursor - 1)
        return self.current

    def redo(self) -> Project:
        """Move to the next snapshot, or keep the latest one at the boundary."""
        self._cursor = min(len(self._states) - 1, self._cursor + 1)
        return self.current
