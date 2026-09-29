"""Exact, inspectable mapping from note-list entries to tablature cells.

Quantization belongs to transcription, never presentation. This layout retains each
stored onset, inserts extra columns for off-grid edits, and assigns rows strictly by
one-based string number. Notes at a shared onset align vertically, while distinct
onsets stay distinct even inside a coarse grid cell. Renderers can use note indices
to prove that every displayed fret came from the current authoritative note list.
"""

from dataclasses import dataclass
from math import floor

from autotab.domain import Project, integer


@dataclass(frozen=True)
class TabBlock:
    """One short tab line with exact beat columns and source note indices by cell."""

    beats: tuple[float, ...]
    cells: dict[tuple[int, int], int]


def layout(project: Project, beats_per_line: int = 4) -> tuple[TabBlock, ...]:
    """Build short lines without re-rounding, reordering strings, or dropping events.

    Dictionary values are source indices, not copied fret values, so text and future
    visual renderers always read the same Note objects as the table. Two simultaneous
    frets on one string are rejected as inconsistent data. Resource bounds protect
    the UI from an accidentally enormous beat position in a manually edited score.
    """
    integer(beats_per_line, "Beats per line", 1, 64)
    last = max((note.beat for note in project.notes), default=0)
    if last * project.subdivisions > 100_000:
        raise ValueError("Text tab exceeds 100,000 grid cells. Use JSON or CSV for this score.")
    # Begin at the measure-sized block containing the first note. A recording may
    # contain several seconds of setup before playing starts, especially when the
    # user leaves "first beat" at zero. Keeping the true beat labels while omitting
    # wholly empty leading blocks makes that score readable without changing time.
    first = min((note.beat for note in project.notes), default=0)
    first_block = floor(first / beats_per_line) * beats_per_line
    blocks = []
    for start in range(
        first_block,
        floor(last / beats_per_line) * beats_per_line + 1,
        beats_per_line,
    ):
        stop = min(start + beats_per_line, floor(last) + 1)
        events = [(i, note) for i, note in enumerate(project.notes) if start <= note.beat < stop]
        grid = project.subdivisions
        beats = tuple(
            sorted(
                {tick / grid for tick in range(start * grid, stop * grid)}
                | {note.beat for _, note in events}
            )
        )
        columns = {beat: column for column, beat in enumerate(beats)}
        cells = {}
        for index, note in events:
            key = (note.string, columns[note.beat])
            if key in cells:
                raise ValueError("Two notes share a string and exact beat. Correct the note list.")
            cells[key] = index
        blocks.append(TabBlock(beats, cells))
    return tuple(blocks)
