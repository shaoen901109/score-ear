# -*- coding: utf-8 -*-
"""suppress_harmonics() 是純函式，不依賴 basic-pitch 模型，獨立成檔案避免被
tests/test_engine_transcribe_poly.py 的 `pytestmark = pytest.mark.slow` 拖累。
"""
from engine.transcribe_poly import suppress_harmonics
from notation.score_model import NoteEvent


def test_suppress_harmonics_removes_octave_low_confidence():
    a = NoteEvent(onset=1.0, offset=1.5, pitch=60, confidence=0.9)
    b = NoteEvent(onset=1.02, offset=1.5, pitch=72, confidence=0.3)  # +12 半音，confidence 低

    result = suppress_harmonics([a, b])

    assert result == [a]


def test_suppress_harmonics_removes_octave_plus_fifth():
    a = NoteEvent(onset=1.0, offset=1.5, pitch=48, confidence=0.8)
    b = NoteEvent(onset=1.01, offset=1.5, pitch=67, confidence=0.2)  # +19 半音

    result = suppress_harmonics([a, b])

    assert result == [a]


def test_suppress_harmonics_keeps_note_when_confidence_close():
    a = NoteEvent(onset=1.0, offset=1.5, pitch=60, confidence=0.6)
    b = NoteEvent(onset=1.01, offset=1.5, pitch=72, confidence=0.5)  # 0.5 >= 0.5*0.6=0.3，保留

    result = suppress_harmonics([a, b])

    assert a in result and b in result


def test_suppress_harmonics_keeps_note_when_onset_far_apart():
    a = NoteEvent(onset=1.0, offset=1.5, pitch=60, confidence=0.9)
    b = NoteEvent(onset=1.2, offset=1.7, pitch=72, confidence=0.1)  # onset 差 200ms

    result = suppress_harmonics([a, b])

    assert a in result and b in result


def test_suppress_harmonics_keeps_non_harmonic_interval():
    a = NoteEvent(onset=1.0, offset=1.5, pitch=60, confidence=0.9)
    b = NoteEvent(onset=1.01, offset=1.5, pitch=64, confidence=0.1)  # +4 半音，不是泛音列間隔

    result = suppress_harmonics([a, b])

    assert a in result and b in result


def test_suppress_harmonics_empty_list():
    assert suppress_harmonics([]) == []


def test_suppress_harmonics_chord_notes_not_removed_by_each_other():
    # 一般和弦音之間音量差不多、間隔常不是泛音列半音差，不該被互相刪除。
    notes = [
        NoteEvent(onset=1.0, offset=1.5, pitch=60, confidence=0.8),
        NoteEvent(onset=1.0, offset=1.5, pitch=64, confidence=0.8),
        NoteEvent(onset=1.0, offset=1.5, pitch=67, confidence=0.8),
    ]
    result = suppress_harmonics(notes)
    assert result == notes
