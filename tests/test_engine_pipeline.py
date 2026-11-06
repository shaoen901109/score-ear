# -*- coding: utf-8 -*-
from engine.pipeline import _dedup_other_vs_vocals
from notation.score_model import NoteEvent


def test_dedup_removes_same_pitch_lower_confidence_duplicate():
    vocal_events = [NoteEvent(onset=1.0, offset=1.5, pitch=67, confidence=0.9)]
    other_events = [NoteEvent(onset=1.02, offset=1.5, pitch=67, confidence=0.4)]

    kept, removed = _dedup_other_vs_vocals(other_events, vocal_events)

    assert removed == 1
    assert kept == []


def test_dedup_removes_octave_duplicate():
    vocal_events = [NoteEvent(onset=2.0, offset=2.4, pitch=60, confidence=0.8)]
    other_events = [NoteEvent(onset=2.03, offset=2.4, pitch=72, confidence=0.3)]

    kept, removed = _dedup_other_vs_vocals(other_events, vocal_events)

    assert removed == 1
    assert kept == []


def test_dedup_keeps_note_when_other_confidence_higher():
    vocal_events = [NoteEvent(onset=1.0, offset=1.5, pitch=67, confidence=0.4)]
    other_events = [NoteEvent(onset=1.02, offset=1.5, pitch=67, confidence=0.9)]

    kept, removed = _dedup_other_vs_vocals(other_events, vocal_events)

    assert removed == 0
    assert kept == other_events


def test_dedup_keeps_note_when_onset_too_far_apart():
    vocal_events = [NoteEvent(onset=1.0, offset=1.5, pitch=67, confidence=0.9)]
    other_events = [NoteEvent(onset=1.2, offset=1.6, pitch=67, confidence=0.3)]

    kept, removed = _dedup_other_vs_vocals(other_events, vocal_events)

    assert removed == 0
    assert kept == other_events


def test_dedup_keeps_note_when_pitch_unrelated():
    vocal_events = [NoteEvent(onset=1.0, offset=1.5, pitch=67, confidence=0.9)]
    other_events = [NoteEvent(onset=1.0, offset=1.5, pitch=68, confidence=0.3)]

    kept, removed = _dedup_other_vs_vocals(other_events, vocal_events)

    assert removed == 0
    assert kept == other_events


def test_dedup_empty_inputs():
    assert _dedup_other_vs_vocals([], []) == ([], 0)
    v = [NoteEvent(onset=0.0, offset=0.1, pitch=60)]
    assert _dedup_other_vs_vocals([], v) == ([], 0)
    o = [NoteEvent(onset=0.0, offset=0.1, pitch=60)]
    kept, removed = _dedup_other_vs_vocals(o, [])
    assert removed == 0
    assert kept == o
