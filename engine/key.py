# -*- coding: utf-8 -*-
"""調性偵測：Krumhansl-Schmuckler 音級輪廓相關法（自製，不依賴 music21）。"""

from __future__ import annotations

from typing import List

import numpy as np

from notation.score_model import KeySignature, NoteEvent

# Krumhansl-Kessler 音級輪廓（index 0 = 主音，往上每格半音）
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# 大調音級（0=C..11=B）對應五度圈 fifths，選擇同音異名中絕對值較小、
# 且與慣用升降記號一致的版本（例如 F# 選 +6 而非 Gb -6）。
_MAJOR_FIFTHS_BY_PC = {
    0: 0,    # C
    1: -5,   # Db
    2: 2,    # D
    3: -3,   # Eb
    4: 4,    # E
    5: -1,   # F
    6: 6,    # F#
    7: 1,    # G
    8: -4,   # Ab
    9: 3,    # A
    10: -2,  # Bb
    11: 5,   # B
}

_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _tonic_name(pc: int, fifths: int) -> str:
    """依 fifths 正負決定用升號還是降號拼寫主音名稱。"""
    pc = pc % 12
    return _SHARP_NAMES[pc] if fifths >= 0 else _FLAT_NAMES[pc]


def _pitch_class_weights(events: List[NoteEvent]) -> np.ndarray:
    """統計各音級（0-11）的加權時值總和，權重為音符時值（秒）。"""
    weights = np.zeros(12, dtype=np.float64)
    for ev in events:
        dur = float(ev.offset) - float(ev.onset)
        if dur <= 0:
            continue
        weights[int(ev.pitch) % 12] += dur
    return weights


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def detect_key(events: List[NoteEvent]) -> KeySignature:
    """用 Krumhansl-Schmuckler 音級輪廓相關法偵測調性。

    權重使用音符時值（秒）；輸出 fifths（-7..7）、mode（major/minor）、
    tonic（依 fifths 正負決定用 # 或 b 拼寫）。沒有可用音符時回傳預設 C 大調。
    """
    weights = _pitch_class_weights(events)
    if not np.any(weights):
        return KeySignature(fifths=0, mode="major", tonic="C")

    best_corr = None
    best_pc = 0
    best_mode = "major"

    for pc in range(12):
        rotated_major = np.roll(_MAJOR_PROFILE, pc)
        rotated_minor = np.roll(_MINOR_PROFILE, pc)
        corr_major = _safe_corr(weights, rotated_major)
        corr_minor = _safe_corr(weights, rotated_minor)

        if best_corr is None or corr_major > best_corr:
            best_corr, best_pc, best_mode = corr_major, pc, "major"
        if best_corr is None or corr_minor > best_corr:
            best_corr, best_pc, best_mode = corr_minor, pc, "minor"

    if best_mode == "major":
        fifths = _MAJOR_FIFTHS_BY_PC[best_pc]
    else:
        # 關係大調 = 小調主音 + 3 半音
        relative_major_pc = (best_pc + 3) % 12
        fifths = _MAJOR_FIFTHS_BY_PC[relative_major_pc]

    tonic = _tonic_name(best_pc, fifths)
    return KeySignature(fifths=fifths, mode=best_mode, tonic=tonic)
