# -*- coding: utf-8 -*-
import pytest

score_model = pytest.importorskip("notation.score_model")

from engine.key import detect_key  # noqa: E402

NoteEvent = score_model.NoteEvent


def _events_from_pitches(pitches):
    events = []
    t = 0.0
    for p in pitches:
        events.append(NoteEvent(onset=t, offset=t + 0.45, pitch=p, velocity=90, confidence=1.0))
        t += 0.5
    return events


def test_detect_key_c_major_scale():
    # C 大調音階 C4..C5，時值當權重
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    key = detect_key(_events_from_pitches(pitches))
    assert key.fifths == 0
    assert key.mode == "major"
    assert key.tonic == "C"


def test_detect_key_empty_defaults_to_c_major():
    key = detect_key([])
    assert key.fifths == 0
    assert key.mode == "major"
    assert key.tonic == "C"
