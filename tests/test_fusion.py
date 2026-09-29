"""Musical cross-matching tests, including ambiguous positions and capos.

Synthetic evidence isolates the multimodal decisions from upstream detector
quality. The output must preserve acoustic pitch even when vision disagrees.
"""

from autotab.domain import AudioNote, Contact, Project, TimedContact
from autotab.fusion import fuse, positions


def test_visible_pitch_match_resolves_same_pitch_on_different_strings():
    """E4 can be played several ways; matching evidence selects B-string fret 5."""
    result = fuse(
        (AudioNote(64, 1, 1.5, 0.95),),
        (TimedContact(Contact(2, 5), 0.9, 1.6),),
        Project("Match", 120),
    )
    assert (result.notes[0].string, result.notes[0].fret) == (2, 5)
    assert result.matched == 1


def test_conflicting_visual_pitch_is_not_trusted():
    """A wrong visible position leaves the acoustic pitch intact and marked for review."""
    result = fuse(
        (AudioNote(64, 0, 1, 0.9),), (TimedContact(Contact(6, 1), 0, 1),), Project("Conflict", 100)
    )
    assert result.notes[0].source == "audio"
    assert (result.notes[0].string, result.notes[0].fret) in positions(64)


def test_nonoverlapping_visual_evidence_cannot_confirm_note():
    """A matching fingering elsewhere in the video is not evidence for this onset."""
    result = fuse(
        (AudioNote(64, 3, 4, 0.9),), (TimedContact(Contact(2, 5), 0, 1),), Project("Timing", 100)
    )
    assert result.matched == 0


def test_repeated_picking_produces_two_notes_despite_one_held_contact():
    """Audio attacks split repeated notes even while the fretting hand stays still."""
    result = fuse(
        (AudioNote(64, 0, 0.4, 0.9), AudioNote(64, 0.5, 0.9, 0.9)),
        (TimedContact(Contact(2, 5), 0, 1),),
        Project("Repeated", 120),
    )
    assert len(result.notes) == 2
    assert result.matched == 2


def test_bass_low_e_and_out_of_range_pitch():
    """Bass low E maps to its open fourth string, while impossible pitches are counted."""
    result = fuse(
        (AudioNote(28, 0, 1, 0.9), AudioNote(10, 1, 2, 0.9)),
        (),
        Project("Bass", 100, instrument="bass"),
    )
    assert (result.notes[0].string, result.notes[0].fret) == (4, 0)
    assert result.out_of_range == 1


def test_capo_uses_relative_tab_frets_and_transposed_audio_pitch():
    """With capo 2, string 2 fret 5 sounds F#4 instead of E4."""
    result = fuse(
        (AudioNote(66, 0, 1, 0.9),),
        (TimedContact(Contact(2, 5), 0, 1),),
        Project("Capo", 100, capo=2),
    )
    assert result.notes[0].fret == 5
    assert result.notes[0].string == 2
    assert result.matched == 1
    assert (6, 0) in positions(42, capo=2)
