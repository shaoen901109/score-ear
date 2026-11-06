# -*- coding: utf-8 -*-
"""測試共用工具：合成音訊（不是測試檔，檔名故意不用 test_ 開頭避免被 pytest 收集）。"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def sine_tone(freq: float, duration_s: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def note_sequence(
    midis: Iterable[float],
    note_duration_s: float,
    sr: int,
    amplitude: float = 0.5,
    attack_decay: bool = False,
) -> np.ndarray:
    """把一串 MIDI 音高合成成正弦波序列（每個音符固定時長，可選擇加上衰減包絡讓 onset 更明顯）。"""
    n_samples = int(sr * note_duration_s)
    chunks = []
    for m in midis:
        tone = sine_tone(midi_to_freq(m), note_duration_s, sr, amplitude)
        if attack_decay:
            n = len(tone)
            env = np.exp(-np.linspace(0, 6, n)).astype(np.float32)
            tone = tone * env
        chunks.append(tone[:n_samples])
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
