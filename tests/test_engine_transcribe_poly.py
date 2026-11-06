# -*- coding: utf-8 -*-
import pytest

pytest.importorskip("basic_pitch")

pytestmark = pytest.mark.slow

from engine.transcribe_poly import _default_thresholds, _midi_to_hz  # noqa: E402


def test_default_thresholds_from_min_note_ms():
    onset, frame, min_len = _default_thresholds(100)
    assert onset == 0.5
    assert frame == 0.3
    assert min_len == 100.0


def test_midi_to_hz_a4():
    assert abs(_midi_to_hz(69) - 440.0) < 1e-6
