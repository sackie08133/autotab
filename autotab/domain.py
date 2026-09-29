"""The shared vocabulary and invariants of a tablature project.

Notes use conventional one-based string numbers and zero-based fret/beat
positions. Durations are measured in beats; observed contacts and audio events
use seconds until quantization. Frozen dataclasses prevent the UI or a detector
from silently mutating an accepted score. Constructors are the validation
boundary for imported JSON, manual edits, and adapter output alike.

Keep this module free of NumPy, OpenCV, MediaPipe, and Streamlit so the musical
data model remains usable in a CLI, another frontend, or a future training tool.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Instrument:
    """Standard tuning ordered from string 1 (highest pitch) to the lowest string.

    MIDI numbers let audio and vision compare exact pitches without relying on
    enharmonic note names. Project.capo transposes these standard open pitches.
    """

    name: str
    open_midi: tuple[int, ...]
    labels: tuple[str, ...]


INSTRUMENTS = {
    "guitar": Instrument(
        "6-string guitar", (64, 59, 55, 50, 45, 40), ("e", "B", "G", "D", "A", "E")
    ),
    "bass": Instrument("4-string bass", (43, 38, 33, 28), ("G", "D", "A", "E")),
}


def finite_number(value: float, name: str, minimum: float = 0) -> None:
    """Reject nonnumeric, nonfinite, boolean, or below-boundary input.

    Explicitly excluding booleans matters because Python otherwise treats them as
    integers. NaN/Infinity would poison timestamp ordering and JSON persistence.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")


def integer(value: int, name: str, minimum: int, maximum: int) -> None:
    """Require an actual integer within an inclusive range, with a readable error.

    Fractional frets and string numbers are never rounded silently. Table adapters
    may convert whole-number floats before constructing a domain object.
    """
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")


@dataclass(frozen=True)
class Calibration:
    """Normalized image corners: near high e, far high e, far low E, near low E.

    Near/far refer to distance from the nut. Corners lie on the outer strings
    at the specified fret wires, with fret 0 representing the nut.
    """

    corners: tuple[tuple[float, float], ...]
    first_fret: int = 0
    last_fret: int = 12

    def __post_init__(self) -> None:
        """Validate wire order and a nondegenerate, convex perimeter of four corners.

        Either orientation is allowed for mirrored/left-handed recordings. Crossing
        or collinear corners would make the perspective transform ambiguous.
        """
        integer(self.first_fret, "First fret wire", 0, 23)
        integer(self.last_fret, "Last fret wire", 1, 24)
        if self.first_fret >= self.last_fret:
            raise ValueError("Last fret wire must be after the first fret wire.")
        if len(self.corners) != 4 or any(len(p) != 2 for p in self.corners):
            raise ValueError("Calibration needs four (x, y) corners.")
        for point in self.corners:
            for value in point:
                finite_number(value, "Corner coordinate")
                if value > 1:
                    raise ValueError("Corner coordinates must be normalized from 0 to 1.")
        crosses = []
        for i in range(4):
            a, b, c = (self.corners[(i + j) % 4] for j in range(3))
            crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not (all(c > 1e-6 for c in crosses) or all(c < -1e-6 for c in crosses)):
            raise ValueError("Corners must form a nonzero convex quadrilateral in perimeter order.")


@dataclass(frozen=True)
class Note:
    """One editable event; provenance distinguishes drafts from human corrections.

    'matched' means audio and visual position agree, not that the note is guaranteed
    correct. The enclosing Project enforces the selected instrument's string count.
    """

    string: int
    fret: int
    beat: float
    duration: float
    source: str = "manual"
    review_reason: str = ""

    def __post_init__(self) -> None:
        """Validate physical positions, nonnegative onset, and positive beat duration."""
        integer(self.string, "String", 1, 6)
        integer(self.fret, "Fret", 0, 24)
        finite_number(self.beat, "Beat")
        finite_number(self.duration, "Duration", 0.001)
        if self.source not in {"manual", "vision", "audio", "matched"}:
            raise ValueError("Unknown note source.")
        if not isinstance(self.review_reason, str):
            raise ValueError("Review reason must be text.")


@dataclass(frozen=True)
class Project:
    """Portable score with explicit user tempo, timing origin, tuning, and calibration.

    BPM is required. This prevents analysis code from inventing a tempo when a user
    has not provided one. The offset maps beat zero to a timestamp in the source video.
    """

    title: str
    bpm: float
    notes: tuple[Note, ...] = ()
    beat_offset_seconds: float = 0.0
    subdivisions: int = 4
    calibration: Calibration | None = None
    instrument: str = "guitar"
    capo: int = 0

    def __post_init__(self) -> None:
        """Check metadata and ensure every note fits the selected instrument."""
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Project title cannot be empty.")
        finite_number(self.bpm, "BPM", 1)
        if self.bpm > 400:
            raise ValueError("BPM must be at most 400.")
        finite_number(self.beat_offset_seconds, "First beat time")
        integer(self.subdivisions, "Subdivisions", 1, 16)
        if any(not isinstance(note, Note) for note in self.notes):
            raise ValueError("Project notes must be Note objects.")
        if self.instrument not in INSTRUMENTS:
            raise ValueError("Instrument must be guitar or bass.")
        integer(self.capo, "Capo fret", 0, 12)
        if any(note.string > len(INSTRUMENTS[self.instrument].open_midi) for note in self.notes):
            raise ValueError("A note refers to a string this instrument does not have.")
        if any(note.fret + self.capo > 24 for note in self.notes):
            raise ValueError("Note exceeds physical fret 24 with the selected capo.")
        if self.calibration and self.calibration.last_fret + self.capo > 24:
            raise ValueError("Calibration exceeds physical fret 24 with the selected capo.")


@dataclass(frozen=True)
class Contact:
    """A proposed finger contact, not a claim that a string was picked."""

    string: int
    fret: int

    def __post_init__(self) -> None:
        """Allow only fretted contacts; an absent finger cannot prove an open string."""
        integer(self.string, "String", 1, 6)
        integer(self.fret, "Fret", 1, 24)


@dataclass(frozen=True)
class TimedContact:
    """A filtered period of visible contact in absolute video seconds."""

    contact: Contact
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        """Reject empty, reversed, or nonfinite observed intervals."""
        finite_number(self.start_seconds, "Start time")
        finite_number(self.end_seconds, "End time")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("A contact must end after it starts.")


@dataclass(frozen=True)
class Fingertip:
    """Continuous fingertip location on the calibrated neck, before hard rounding.

    String 1.4 lies between strings 1 and 2. Fret position 4.8 lies between wires
    4 and 5, where a finger normally sounds fret 5. Keeping these values lets audio
    disambiguate near-string/near-wire detections that a hard threshold would discard.
    These coordinates describe projected landmarks, not confirmed string pressure.
    """

    string_position: float
    fret_position: float

    def __post_init__(self) -> None:
        """Reject nonfinite coordinates before proximity scoring."""
        finite_number(self.string_position, "Projected string")
        finite_number(self.fret_position, "Projected fret")


@dataclass(frozen=True)
class VisualSample:
    """All usable fingertips from one selected fretting hand at an absolute video time."""

    seconds: float
    fingertips: tuple[Fingertip, ...]
    calibration: Calibration | None = None

    def __post_init__(self) -> None:
        """Keep visual evidence time ordered by the pipeline's timestamp contract."""
        finite_number(self.seconds, "Visual sample time")


@dataclass(frozen=True)
class AudioNote:
    """One acoustic pitch interval and its uncalibrated periodicity score.

    MIDI encodes pitch, not instrument position. Fusion chooses among the playable
    string/fret combinations only after this audio observation has been produced.
    """

    midi: int
    start_seconds: float
    end_seconds: float
    quality: float

    def __post_init__(self) -> None:
        """Enforce valid MIDI pitch, positive time interval, and quality from zero to one."""
        integer(self.midi, "MIDI pitch", 0, 127)
        finite_number(self.start_seconds, "Audio start")
        finite_number(self.end_seconds, "Audio end")
        finite_number(self.quality, "Pitch quality")
        if self.end_seconds <= self.start_seconds or self.quality > 1:
            raise ValueError("Invalid audio note duration or quality.")
