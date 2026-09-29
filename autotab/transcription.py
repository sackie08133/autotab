"""Convert noisy per-frame contacts into stable, beat-aligned visual drafts.

ContactTracker is a small temporal state machine: contacts must appear multiple
times, short dropouts get a grace period, and a new fret on the same string
replaces the old one. Quantization is a separate pure operation using the user's
BPM and first-beat timestamp. This layer contains no image or sound processing.

In audiovisual mode the tracker supplies evidence to fusion; its finger dwell
durations do not become acoustic note timings. Visual-only mode remains useful
for silent clips, with explicit limitations around repeated picking and rests.
"""

from dataclasses import dataclass
from math import floor

from autotab.domain import Contact, Note, Project, TimedContact, finite_number, integer


@dataclass
class _Track:
    """Mutable internal evidence accumulator for one currently visible contact."""

    first: float
    last: float
    observations: int = 1


class ContactTracker:
    """Suppress flicker and bridge brief occlusions; only one fret per string survives.

    A higher fret masks lower contacts on the same string. Duration here is finger
    dwell time, not acoustic sustain. A held finger cannot reveal repeated picking.
    """

    def __init__(self, min_observations: int = 2, release_seconds: float = 0.12):
        """Set the confirmation count and tolerated gap, both independent of BPM."""
        integer(min_observations, "Minimum observations", 1, 100)
        finite_number(release_seconds, "Release grace")
        self.min_observations = min_observations
        self.release_seconds = release_seconds
        self._active: dict[Contact, _Track] = {}
        self._finished: list[TimedContact] = []
        self._last_time = -1.0

    def update(self, seconds: float, contacts: tuple[Contact, ...]) -> None:
        """Consume one strictly later frame, opening/updating/closing contact tracks.

        Higher frets mask lower fingers on the same string. A replacement closes
        immediately; ordinary missing evidence waits for the configured grace.
        """
        finite_number(seconds, "Frame timestamp")
        if seconds <= self._last_time:
            raise ValueError("Frame timestamps must increase strictly.")
        self._last_time = seconds
        by_string: dict[int, Contact] = {}
        for contact in contacts:
            if contact.string not in by_string or contact.fret > by_string[contact.string].fret:
                by_string[contact.string] = contact
        present = set(by_string.values())
        for contact, track in list(self._active.items()):
            replaced = contact.string in by_string and contact not in present
            expired = seconds - track.last > self.release_seconds
            if replaced or expired:
                # End at the replacement frame, or the end of the occlusion grace.
                end = seconds if replaced else min(seconds, track.last + self.release_seconds)
                self._close(contact, end)
        for contact in present:
            if contact in self._active:
                self._active[contact].last = seconds
                self._active[contact].observations += 1
            else:
                self._active[contact] = _Track(seconds, seconds)

    def _close(self, contact: Contact, end: float) -> None:
        """Remove an active track, emitting it only if sufficient evidence accumulated."""
        track = self._active.pop(contact)
        if track.observations >= self.min_observations and end > track.first:
            self._finished.append(TimedContact(contact, track.first, end))

    def finish(self, end_seconds: float) -> tuple[TimedContact, ...]:
        """Flush remaining tracks at end of video and return chronologically sorted evidence.

        The caller includes the final sample interval in end_seconds. Missing
        contacts are never extended beyond their last observation plus grace.
        """
        finite_number(end_seconds, "Video end")
        if end_seconds < self._last_time:
            raise ValueError("Video end precedes the last frame.")
        for contact, track in list(self._active.items()):
            self._close(contact, min(end_seconds, track.last + self.release_seconds))
        return tuple(
            sorted(self._finished, key=lambda event: (event.start_seconds, event.contact.string))
        )


def quantize(events: tuple[TimedContact, ...], settings: Project) -> tuple[Note, ...]:
    """Snap to the user's beat grid. Beat zero is the user-entered video offset."""
    scale = settings.bpm / 60
    grid = settings.subdivisions
    notes: dict[tuple[int, float], Note] = {}
    for event in events:
        start = (event.start_seconds - settings.beat_offset_seconds) * scale
        end = (event.end_seconds - settings.beat_offset_seconds) * scale
        if end <= 0:
            continue
        beat = floor(max(0, start) * grid + 0.5) / grid
        end_beat = floor(end * grid + 0.5) / grid
        note = Note(
            event.contact.string, event.contact.fret, beat, max(1 / grid, end_beat - beat), "vision"
        )
        # Two fast transitions may snap onto the same string/grid cell. Keep the later one.
        notes[(note.string, note.beat)] = note
    return tuple(sorted(notes.values(), key=lambda note: (note.beat, note.string)))
