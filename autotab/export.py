"""Presentation exports derived from the authoritative beat-based score.

CSV preserves event timing and provenance for inspection or downstream tools.
Fixed-width text tablature shows every stored start without re-quantizing it and
labels the highest string first. It does not invent rhythmic notation or imply
that hyphens encode note duration.
JSON in storage.py is the format for saving and reopening the whole project.
"""

import csv
import io

from autotab.domain import INSTRUMENTS, Project
from autotab.score import rhythm
from autotab.tablature import layout


def to_csv(project: Project) -> str:
    """Export sorted events with beat durations and video-relative onset seconds.

    CSV quoting is handled by the standard library; timing derives only from the
    project BPM and offset. Instrument metadata remains in the full JSON project.
    """
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["string", "fret", "beat", "duration_beats", "start_seconds", "source"])
    for note in sorted(project.notes, key=lambda n: (n.beat, n.string)):
        writer.writerow(
            [
                note.string,
                note.fret,
                note.beat,
                note.duration,
                project.beat_offset_seconds + note.beat * 60 / project.bpm,
                note.source,
            ]
        )
    return output.getvalue()


def to_ascii(project: Project, beats_per_line: int = 4) -> str:
    """Beat-aligned fixed-width tab; durations remain in the project and CSV.

    Every stored onset has its own column, including off-grid manual edits.
    Presentation never requantizes the note list. Short lines make all six/four string
    rows readable without the old 256-character horizontal scroll by default.
    """
    grid = project.subdivisions
    blocks = layout(project, beats_per_line)
    labels = INSTRUMENTS[project.instrument].labels
    lines = [
        f"{project.title} | {project.bpm:g} BPM | {grid} cells/beat | Capo {project.capo}",
        "Strings (top to bottom): "
        + ", ".join(f"{i}={label}" for i, label in enumerate(labels, 1)),
        "4/4 bars. Rhythm below: w=whole h=half q=quarter e=eighth s=sixteenth t=32nd.",
        "A dot adds half the value; b means an exact duration in beats.",
        "",
    ]
    for block in blocks:
        beat_labels = [f"{beat:.9g}" for beat in block.beats]
        if len(set(beat_labels)) != len(beat_labels):
            beat_labels = [repr(beat) for beat in block.beats]
        widths = [max(4, len(label) + 1) for label in beat_labels]
        # Keep timing columns internally for exact alignment, but label only bar
        # landmarks above the strings. The table remains the place to edit beats.
        landmark = "Bar " + str(int(block.beats[0] // 4) + 1)
        lines.append("   " + landmark)
        for string, label in enumerate(labels, start=1):
            line = "".join(
                str(project.notes[block.cells[(string, column)]].fret).ljust(width, "-")
                if (string, column) in block.cells
                else "-" * width
                for column, width in enumerate(widths)
            )
            lines.append(f"{label}| {line}|")
        durations = [
            sorted(
                {
                    project.notes[index].duration
                    for (_, col), index in block.cells.items()
                    if col == column
                },
                reverse=True,
            )
            for column in range(len(block.beats))
        ]
        for lane in range(max((len(values) for values in durations), default=0)):
            lines.append(
                "   "
                + "".join(
                    (rhythm(values[lane]).code if lane < len(values) else "").ljust(width)
                    for values, width in zip(durations, widths, strict=True)
                ).rstrip()
            )
        lines.append("")
    return "\n".join(lines)
