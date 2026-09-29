"""Bar-based SVG tablature with a separate, aligned rhythm staff below the strings.

The score is a presentation of immutable notes, never a second transcription pass.
One beat remains a quarter note and this first engraver uses explicit 4/4 bars.
Onsets are positioned within numbered bars; each bar has enough space to separate
nearby edited notes. No beat ruler is drawn. Standard durations use vector noteheads,
stems, flags/beams, and dots, avoiding dependence on installed music fonts. Unusual
durations retain an exact beat-length label instead of being silently rounded.

Notes spanning barlines are split only for display, with parenthesized continuation
frets and ties. Original note indices remain on SVG groups for traceability. Chords
share rhythm marks when their onset and duration agree; independent durations get
separate rhythm lanes. The existing note table and synth keep the original timing.
All user-controlled text is escaped before inclusion in the standalone SVG.
"""

from dataclasses import dataclass
from html import escape
from math import ceil, floor, isclose

from autotab.domain import INSTRUMENTS, Project

BAR_BEATS = 4
INK = "#e5e7eb"
BACKGROUND = "#1c1e22"


@dataclass(frozen=True)
class Rhythm:
    """Recognized duration in quarter-note beats, plus its engraving instructions."""

    name: str
    code: str
    hollow: bool = False
    stem: bool = True
    flags: int = 0
    dotted: bool = False
    exact: bool = True


def rhythm(duration: float) -> Rhythm:
    """Recognize standard or dotted durations without quantizing arbitrary edits.

    Only floating-point arithmetic noise is tolerated. A duration such as 0.9 beats
    must not falsely appear as a quarter note; its exact length is labeled instead.
    """
    for beats, name, code, flags in (
        (4, "whole", "w", 0),
        (2, "half", "h", 0),
        (1, "quarter", "q", 0),
        (0.5, "eighth", "e", 1),
        (0.25, "sixteenth", "s", 2),
        (0.125, "thirty-second", "t", 3),
    ):
        for dotted in (False, True):
            if isclose(duration, beats * (1.5 if dotted else 1), rel_tol=1e-9, abs_tol=1e-9):
                return Rhythm(
                    ("dotted " if dotted else "") + name,
                    code + ("." if dotted else ""),
                    beats >= 2,
                    beats < 4,
                    flags,
                    dotted,
                )
    return Rhythm(f"{duration:g} beats", f"{duration:g}b", exact=False)


@dataclass(frozen=True)
class Segment:
    """A source note's visible intersection with one bar, with tie status at each end."""

    index: int
    beat: float
    duration: float
    continued: bool
    continues: bool


def bars(project: Project) -> tuple[tuple[int, tuple[Segment, ...]], ...]:
    """Partition a score into numbered 4/4 bars, including sustain-only bars.

    Empty leading bars are omitted while their absolute numbering is preserved.
    The 1,000-bar bound prevents an accidental huge edit from creating an enormous
    DOM; it never changes the saved project or the available JSON/CSV exports.
    """
    first = floor(min((n.beat for n in project.notes), default=0) / BAR_BEATS)
    last = max(first, ceil(max((n.beat + n.duration for n in project.notes), default=4) / 4) - 1)
    if last - first >= 1000:
        raise ValueError(
            "Score preview is limited to 1,000 bars. Export JSON or CSV for this score."
        )
    occupied = set()
    for note in project.notes:
        key = (note.string, note.beat)
        if key in occupied:
            raise ValueError("Two notes share a string and exact beat. Correct the note list.")
        occupied.add(key)
    result = []
    for bar in range(first, last + 1):
        start, end = bar * 4, (bar + 1) * 4
        segments = []
        for index, note in enumerate(project.notes):
            begin, stop = max(start, note.beat), min(end, note.beat + note.duration)
            if stop > begin:
                segments.append(
                    Segment(
                        index,
                        begin,
                        stop - begin,
                        note.beat < start,
                        note.beat + note.duration > end,
                    )
                )
        result.append((bar + 1, tuple(sorted(segments, key=lambda s: (s.beat, s.index)))))
    return tuple(result)


def _text(x: float, y: float, value: str, size: int = 14, anchor: str = "middle") -> str:
    """Escape SVG text at the final rendering boundary, including project titles."""
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size}" text-anchor="{anchor}">'
        f"{escape(value)}</text>"
    )


def _line(x1: float, y1: float, x2: float, y2: float, width: float = 1.5) -> str:
    """Draw a vector stroke using the score's inherited current text color."""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="{INK}" stroke-width="{width:g}"/>'
    )


def _mark(x: float, y: float, value: Rhythm, beamed: bool = False) -> str:
    """Draw a downward-stem rhythm symbol beneath a fret column.

    Whole/half notes have hollow heads; quarter and shorter notes are filled.
    Shared beams replace flags only when the caller has verified adjacent timings.
    Nonstandard durations use a small numeric label rather than a misleading glyph.
    """
    if not value.exact:
        return _text(x, y + 12, value.code, 11)
    parts = [f'<g class="rhythm" data-duration="{value.name}"><title>{value.name} note</title>']
    parts.append(
        f'<ellipse cx="{x:g}" cy="{y:g}" rx="6" ry="4" '
        f'fill="{BACKGROUND if value.hollow else INK}" stroke="{INK}" '
        'stroke-width="1.8"/>'
    )
    if value.stem:
        parts.append(_line(x - 5, y, x - 5, y + 30))
    if not beamed:
        for flag in range(value.flags):
            fy = y + 30 - flag * 7
            parts.append(
                f'<path d="M {x - 5:g} {fy:g} q 14 -7 9 -18" '
                f'fill="none" stroke="{INK}" stroke-width="2.5"/>'
            )
    if value.dotted:
        parts.append(f'<circle cx="{x + 12:g}" cy="{y:g}" r="2"/>')
    return "".join(parts) + "</g>"


def _bar_svg(project: Project, number: int, segments: tuple[Segment, ...]) -> tuple[str, int, int]:
    """Render one complete bar, returning markup, width, and height for row layout.

    Proportional spacing is increased only where needed to avoid overlapping fret
    text. A rhythm lane is allocated for each different duration at a shared onset.
    Adjacent short notes with identical flag counts beam within a quarter-note beat.
    """
    start = (number - 1) * 4
    onsets = sorted({s.beat for s in segments})
    xs = {}
    previous_x, previous_beat = 38.0, float(start)
    for beat in onsets:
        x = max(38 + (beat - start) * 85, previous_x + (44 if xs else 0))
        xs[beat] = x
        previous_x, previous_beat = x, beat
    width = ceil(max(390, previous_x + (start + 4 - previous_beat) * 85 + 24))
    count = len(INSTRUMENTS[project.instrument].labels)
    bottom = 45 + (count - 1) * 20
    parts = [_text(12, 20, str(number), 14, "start")]
    for string in range(count):
        parts.append(_line(0, 45 + string * 20, width, 45 + string * 20, 0.65))
    parts.extend((_line(0, 45, 0, bottom), _line(width, 45, width, bottom)))
    groups = {}
    for segment in segments:
        note = project.notes[segment.index]
        x, y = xs[segment.beat], 45 + (note.string - 1) * 20
        fret = f"({note.fret})" if segment.continued else str(note.fret)
        label = (
            f"String {note.string}, fret {note.fret}, beat {note.beat:g}, {note.duration:g} beats"
        )
        parts.append(
            f'<g class="tab-note" data-note-index="{segment.index}" '
            f'data-beat="{segment.beat:g}" data-string="{note.string}">'
            f'<title>{label}</title><rect x="{x - len(fret) * 5 - 3:g}" y="{y - 10:g}" '
            f'width="{len(fret) * 10 + 6}" height="19" fill="{BACKGROUND}"/>'
            + _text(x, y + 6, fret, 18)
        )
        if segment.continued:
            parts.append(
                f'<path class="tie" d="M 5 {y + 7} Q {x / 2:g} {y + 20} {x - 12:g} {y + 7}" '
                f'fill="none" stroke="{INK}"/>'
            )
        if segment.continues:
            parts.append(
                f'<path class="tie" d="M {x + 12:g} {y + 7} '
                f'Q {(x + width) / 2:g} {y + 20} {width - 5} {y + 7}" '
                f'fill="none" stroke="{INK}"/>'
            )
        parts.append("</g>")
        groups.setdefault(segment.beat, set()).add(segment.duration)
    lanes = max((len(durations) for durations in groups.values()), default=1)
    for lane in range(lanes):
        events = [
            (beat, sorted(groups[beat], reverse=True)[lane])
            for beat in onsets
            if len(groups[beat]) > lane
        ]
        links = set()
        for i, ((beat, duration), (next_beat, next_duration)) in enumerate(
            zip(events, events[1:], strict=False)
        ):
            value, following = rhythm(duration), rhythm(next_duration)
            if (
                value.flags > 0
                and value.flags == following.flags
                and floor(beat + 1e-9) == floor(next_beat + 1e-9)
                and isclose(beat + duration, next_beat, abs_tol=1e-9)
            ):
                links.add(i)
                for flag in range(value.flags):
                    y = bottom + 62 + lane * 54 - flag * 7
                    parts.append(
                        '<g class="beam">'
                        + _line(xs[beat] - 5, y, xs[next_beat] - 5, y, 4)
                        + "</g>"
                    )
        for i, (beat, duration) in enumerate(events):
            parts.append(
                _mark(
                    xs[beat],
                    bottom + 32 + lane * 54,
                    rhythm(duration),
                    i in links or i - 1 in links,
                )
            )
    if not segments:
        parts.append(_text(width / 2, bottom + 40, "Rest — whole bar", 13))
    return "".join(parts), width, bottom + 85 + (lanes - 1) * 54


def to_svg(project: Project) -> str:
    """Create a standalone dark score with two bars per line and a duration legend.

    SVG is composed from validated score values and escaped strings. It is usable
    in the web app and as a downloadable vector file, with no script or network
    dependencies. Bar labels are one-based; the editable table remains zero-based.
    """
    rendered = [_bar_svg(project, number, segments) for number, segments in bars(project)]
    parts = [
        _text(24, 32, project.title, 20, "start"),
        _text(24, 58, f"4/4 · {project.bpm:g} BPM · Capo {project.capo}", 14, "start"),
    ]
    y, total_width = 80, 880
    for offset in range(0, len(rendered), 2):
        row = rendered[offset : offset + 2]
        x = 42
        for string, label in enumerate(INSTRUMENTS[project.instrument].labels):
            parts.append(_text(23, y + 51 + string * 20, label, 13))
        for svg, width, _ in row:
            parts.append(f'<g transform="translate({x},{y})">{svg}</g>')
            x += width
        total_width = max(total_width, x + 24)
        y += max(height for _, _, height in row) + 16
    parts.append(_text(24, y + 10, "Rhythm key", 13, "start"))
    for index, duration in enumerate((4, 2, 1, 0.5, 0.25, 0.125)):
        x = 65 + index * 135
        parts.append(_mark(x, y + 36, rhythm(duration)))
        parts.append(_text(x + 15, y + 42, rhythm(duration).name, 12, "start"))
    parts.append(
        _text(
            24,
            y + 95,
            "Dot = + half the value · Curves / (frets) = tied continuation · b = beats",
            12,
            "start",
        )
    )
    height = y + 116
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{height}" '
        f'viewBox="0 0 {total_width} {height}" role="img" aria-label="Tablature with rhythm" '
        f'font-family="Arial, sans-serif" fill="{INK}">'
        f'<rect width="100%" height="100%" rx="8" fill="{BACKGROUND}"/>' + "".join(parts) + "</svg>"
    )
