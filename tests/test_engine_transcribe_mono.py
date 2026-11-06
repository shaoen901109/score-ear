# -*- coding: utf-8 -*-
import pytest

score_model = pytest.importorskip("notation.score_model")

from engine.transcribe_mono import monophonize  # noqa: E402

NoteEvent = score_model.NoteEvent


def test_monophonize_trims_lower_confidence_overlap():
    low = NoteEvent(onset=0.0, offset=1.0, pitch=60, velocity=80, confidence=0.3)
    high = NoteEvent(onset=0.4, offset=0.6, pitch=64, velocity=90, confidence=0.9)

    result = monophonize([low, high])

    # 高信心音符本身要完整保留
    assert any(ev.pitch == 64 and ev.onset == 0.4 and ev.offset == 0.6 for ev in result)
    # 低信心音符在被 high 覆蓋的 [0.4, 0.6) 範圍內不應該有殘留
    for ev in result:
        if ev.pitch == 60:
            assert not (ev.onset < 0.6 and ev.offset > 0.4)
    # 時間軸上任兩個保留的音符不應互相重疊（單音化）
    ordered = sorted(result, key=lambda e: e.onset)
    for a, b in zip(ordered, ordered[1:]):
        assert a.offset <= b.onset + 1e-9


def test_monophonize_merges_adjacent_same_pitch():
    a = NoteEvent(onset=0.0, offset=0.5, pitch=60, velocity=80, confidence=1.0)
    b = NoteEvent(onset=0.51, offset=1.0, pitch=60, velocity=80, confidence=1.0)

    result = monophonize([a, b], merge_gap_s=0.05)

    assert len(result) == 1
    assert result[0].onset == 0.0
    assert result[0].offset == 1.0


def test_monophonize_empty_input():
    assert monophonize([]) == []
