# -*- coding: utf-8 -*-
import json

import numpy as np
import pytest

from engine.transcribe_drums import (
    HIHAT_PITCH,
    KICK_PITCH,
    SNARE_PITCH,
    classify_window,
    transcribe_drums,
)


def test_classify_window_low_freq_pulse_is_kick():
    sr = 22050
    n = int(sr * 0.05)
    t = np.linspace(0, 0.05, n, endpoint=False)
    y = 0.9 * np.sin(2 * np.pi * 60.0 * t)  # 60Hz 低頻脈衝

    pitches = classify_window(y.astype(np.float32), sr)

    assert KICK_PITCH in pitches


def test_classify_window_high_freq_noise_is_hihat():
    sr = 22050
    n = int(sr * 0.05)
    t = np.linspace(0, 0.05, n, endpoint=False)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(6000.0, 10000.0, size=8)
    y = np.zeros(n, dtype=np.float64)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t)
    y = (y / np.max(np.abs(y))) * 0.9  # 高頻噪音脈衝（多個高頻正弦疊加模擬）

    pitches = classify_window(y.astype(np.float32), sr)

    assert HIHAT_PITCH in pitches


def test_classify_window_empty_returns_default():
    pitches = classify_window(np.array([], dtype=np.float32), 22050)
    assert pitches == [KICK_PITCH]


def test_classify_window_snare_tone_plus_noise_is_snare():
    sr = 22050
    n = int(sr * 0.08)
    t = np.arange(n, dtype=np.float64) / sr
    tone = np.sin(2 * np.pi * 200.0 * t) * np.exp(-t * 25.0)  # 200Hz 音頭
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = spectrum * ((freqs >= 1000.0) & (freqs < 4000.0))
    noise_band = np.fft.irfft(spectrum, n=n) * np.exp(-t * 18.0)
    y = 0.45 * tone + 0.75 * (noise_band / (np.max(np.abs(noise_band)) or 1.0))
    y = (y / (np.max(np.abs(y)) or 1.0)) * 0.8

    pitches = classify_window(y.astype(np.float32), sr)

    assert SNARE_PITCH in pitches
    assert KICK_PITCH not in pitches


# ---------------------------------------------------------------------------
# 端到端：multi_mix 鼓聲 fixture（kick/snare/hihat 各自 F1 >= 0.8）
# ---------------------------------------------------------------------------

librosa = pytest.importorskip("librosa")


def _onset_f1(gt_notes, pred_events, pitch, tol=0.05):
    gt = [n for n in gt_notes if n["pitch"] == pitch]
    pred = [e for e in pred_events if e.pitch == pitch]
    used = [False] * len(pred)
    tp = 0
    for g in sorted(gt, key=lambda d: d["onset"]):
        best_i, best_dt = None, None
        for i, p in enumerate(pred):
            if used[i]:
                continue
            dt = abs(p.onset - g["onset"])
            if dt <= tol and (best_dt is None or dt < best_dt):
                best_dt, best_i = dt, i
        if best_i is not None:
            used[best_i] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    recall = tp / len(gt) if gt else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1


def test_transcribe_drums_multi_mix_fixture_per_instrument_f1():
    """在 tests/make_fixtures.py 的 multi_mix 鼓聲（未經 demucs）上，
    kick/snare/hihat 各自 onset-level F1 都要 >= 0.8（見 docs/ACCURACY.md）。
    """
    import soundfile as sf

    from tests import make_fixtures

    fixtures = make_fixtures.generate_all()
    gt_notes = json.loads(fixtures["multi_mix"]["json"].read_text(encoding="utf-8"))["tracks"]["drums"]
    y, sr = sf.read(str(fixtures["multi_mix_drums"]["wav"]))

    events = transcribe_drums(y.astype(np.float32), sr, min_note_ms=20)

    for pitch, name in ((KICK_PITCH, "kick"), (SNARE_PITCH, "snare"), (HIHAT_PITCH, "hihat")):
        f1 = _onset_f1(gt_notes, events, pitch)
        assert f1 >= 0.8, f"{name} F1={f1:.2f} 低於 0.8"
