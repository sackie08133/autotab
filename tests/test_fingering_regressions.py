"""Regression cases based on the user's Mary Had a Little Lamb feedback.

In standard tuning, G-string frets 9/7/5 sound E4/D4/C4. The original hard-contact
pipeline could discard fingertips between strings, then invent low-position e/B
fingerings for these correct pitches. These tests retain continuous visual evidence,
check attack-local selection, and prove that tab cells map one-to-one to list entries.
Synthetic projected landmarks isolate matching logic; they do not establish real-video
hand-landmark accuracy, which still depends on the recording and calibration.
"""

import pytest

from autotab.adapters.geometry import FretboardMapper
from autotab.domain import (
    AudioNote,
    Calibration,
    Contact,
    Fingertip,
    Note,
    Project,
    TimedContact,
    VisualSample,
)
from autotab.export import to_ascii
from autotab.fusion import fuse
from autotab.review import midi_pitch, pitch_name
from autotab.tablature import layout


def test_mary_phrase_stays_on_g_string_despite_near_string_landmarks():
    """Audio-compatible positions must follow the visible G-string 9–7–5 phrase."""
    frets = (9, 7, 5, 7, 9, 9, 9)
    audio = tuple(
        AudioNote(55 + fret, i * 0.6, i * 0.6 + 0.5, 0.98) for i, fret in enumerate(frets)
    )
    # String 3.4 is outside the previous 0.325 spacing contact cutoff, but provides
    # far better evidence for the G string than any low-fret B/e pitch alternative.
    samples = tuple(
        VisualSample(i * 0.6 + delta, (Fingertip(3.4, fret - 0.2),))
        for i, fret in enumerate(frets)
        for delta in (0.03, 0.10, 0.17)
    )
    result = fuse(audio, (), Project("Mary", 100), visual_samples=samples)
    assert [(n.string, n.fret) for n in result.notes] == [(3, f) for f in frets]
    assert result.matched == len(frets)
    project = Project("Mary", 100, result.notes)
    assert [pitch_name(n, project) for n in result.notes[:3]] == ["E4", "D4", "C4"]
    tab = to_ascii(project)
    assert "G| 9---" in tab
    assert "e| 0" not in tab


def test_near_wire_uncertainty_uses_visible_region_before_open_position():
    """Imprecise wire placement is flagged, but does not force a mid-neck note to open e."""
    result = fuse(
        (AudioNote(64, 0, 0.5, 0.98),),
        (),
        Project("Near wire", 100),
        visual_samples=(VisualSample(0.05, (Fingertip(3.1, 9.5),)),),
    )
    assert (result.notes[0].string, result.notes[0].fret) == (3, 9)
    assert result.notes[0].source == "audio"
    assert result.notes[0].review_reason


@pytest.mark.parametrize(
    "instrument,capo,midi", [("guitar", 0, 64), ("guitar", 2, 66), ("bass", 0, 42)]
)
def test_general_hand_region_outweighs_a_stray_exact_contact(instrument, capo, midi):
    """A misleading precise contact must not beat the bulk of the visible hand.

    Guitar E4 has alternatives B5 and G9; bass F#2 has alternatives A9 and E14.
    In each case an exact landmark points to the distant alternative while three
    other fingertips establish fret 9 as the general playing region. Their string
    coordinates deliberately miss the correct string, so region evidence must win.
    """
    wrong_string, wrong_fret = (4, 14) if instrument == "bass" else (2, 5)
    settings = Project("Region first", 120, instrument=instrument, capo=capo)
    tips = (
        Fingertip(wrong_string, wrong_fret - 0.2),
        Fingertip(1, 7.8),
        Fingertip(1, 8.5),
        Fingertip(1, 9.2),
    )
    audio = (AudioNote(midi, 1, 1.5, 0.98),)
    result = fuse(
        audio,
        (TimedContact(Contact(wrong_string, wrong_fret), 0.9, 1.5),),
        settings,
        visual_samples=tuple(VisualSample(t, tips) for t in (1.02, 1.1, 1.2)),
    )
    note = result.notes[0]
    assert (note.string, note.fret) == (3, 9)
    assert midi_pitch(note, settings) == midi
    assert (note.beat, note.duration) == (2, 1)
    assert note.source == "audio"
    assert "general hand placement" in note.review_reason


def test_general_hand_region_selects_mary_fingerings_without_correct_string_landmarks():
    """The 9–7–5 phrase can be located even when vision gets every string wrong."""
    frets = (9, 7, 5)
    settings = Project("Coarse hand", 60)
    audio = tuple(AudioNote(55 + f, i, i + 0.5, 0.98) for i, f in enumerate(frets))
    samples = tuple(
        VisualSample(i + 0.1, (Fingertip(6, f - 0.8), Fingertip(5, f - 0.2)))
        for i, f in enumerate(frets)
    )
    notes = fuse(audio, (), settings, visual_samples=samples).notes
    assert [(n.string, n.fret) for n in notes] == [(3, f) for f in frets]
    assert [midi_pitch(n, settings) for n in notes] == [a.midi for a in audio]
    assert all(n.source == "audio" for n in notes)


def test_attack_evidence_beats_later_preparatory_hand_movement():
    """A long ringing note must not inherit a different fingering prepared later."""
    visual = (TimedContact(Contact(3, 9), 0, 0.2), TimedContact(Contact(2, 5), 0.4, 2))
    result = fuse((AudioNote(64, 0.05, 2, 0.98),), visual, Project("Ring", 100))
    assert (result.notes[0].string, result.notes[0].fret) == (3, 9)


def test_ambiguous_visual_candidates_remain_flagged():
    """Two equally plausible fingers must not receive a false confirmed label."""
    samples = (VisualSample(0.1, (Fingertip(3, 8.8), Fingertip(2, 4.8))),)
    result = fuse(
        (AudioNote(64, 0, 0.5, 0.98),), (), Project("Ambiguous", 100), visual_samples=samples
    )
    assert result.notes[0].source == "audio"
    assert "ambiguous" in result.notes[0].review_reason


def test_projection_preserves_fingertips_rejected_by_hard_contact_filter():
    """The adapter boundary must retain the evidence the new matcher depends on."""
    mapper = FretboardMapper(Calibration(((0, 0), (1, 0), (1, 1), (0, 1))))
    along = mapper.boundaries[8] * 0.2 + mapper.boundaries[9] * 0.8
    x, y = mapper.image_point(along, (3.4 - 1) / 5)
    assert mapper.contact(x, y) is None
    tip = mapper.fingertip(x, y)
    assert tip.string_position == pytest.approx(3.4)
    assert tip.fret_position == pytest.approx(8.8)


def test_every_tab_cell_preserves_exact_note_index_string_fret_and_beat():
    """Tab rendering cannot reassign G-string notes or re-round a manually corrected beat."""
    notes = (
        Note(3, 9, 0, 1),
        Note(3, 7, 0.13, 0.5),
        Note(3, 5, 1.75, 0.5),
        Note(6, 12, 4.25, 1),
        Note(1, 3, 4.25, 1),
    )
    project = Project("Exact tab", 100, notes)
    recovered = []
    for block in layout(project):
        for (string, column), index in block.cells.items():
            note = project.notes[index]
            assert string == note.string
            assert block.beats[column] == note.beat
            recovered.append(index)
    assert sorted(recovered) == list(range(len(notes)))
    assert "Bar 1" in to_ascii(project)
    assert "0.13" not in to_ascii(project)


def test_tab_omits_empty_leading_blocks_without_changing_the_real_beat():
    """Setup time before the performance must not create pages of blank tab."""
    project = Project("Late start", 100, (Note(3, 9, 4.75, 1),))
    blocks = layout(project)
    assert len(blocks) == 1
    assert blocks[0].beats[0] == 4
    assert blocks[0].beats[3] == 4.75
    assert project.notes[blocks[0].cells[(3, 3)]].beat == 4.75


def test_rapid_repeated_audio_notes_are_not_silently_deleted():
    """Two attacks in one quantization cell survive as distinct tab events."""
    settings = Project("Fast", 60)
    result = fuse((AudioNote(64, 0, 0.05, 0.98), AudioNote(64, 0.06, 0.12, 0.98)), (), settings)
    assert len(result.notes) == 2
    assert result.notes[0].beat != result.notes[1].beat
    assert [midi_pitch(n, settings) for n in result.notes] == [64, 64]
