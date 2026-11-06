# -*- coding: utf-8 -*-
"""通用複音轉譜：使用 basic-pitch predict。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

from notation.score_model import NoteEvent


class PolyTranscribeError(RuntimeError):
    """複音轉譜失敗（附中文說明）。"""


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _default_thresholds(min_note_ms: int):
    """依 min_note_ms 推導 onset_threshold / frame_threshold / minimum_note_length（毫秒）。"""
    onset_threshold = 0.5
    frame_threshold = 0.3
    minimum_note_length = max(float(min_note_ms), 30.0)
    return onset_threshold, frame_threshold, minimum_note_length


def transcribe_poly(
    y: np.ndarray,
    sr: int,
    min_note_ms: int = 60,
    min_pitch: Optional[int] = None,
    max_pitch: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[NoteEvent]:
    """用 basic-pitch 辨識複音音符。

    回傳 NoteEvent 清單（秒為單位，confidence = 音符振幅 amplitude）。
    min_pitch/max_pitch 為 MIDI 音高範圍限制（例如 bass 用 28-60），
    會同時傳給 basic-pitch 的頻率範圍參數，並在輸出時再做一次保險過濾。
    """
    if progress:
        progress(0.0, "載入複音轉譜模型 (basic-pitch)")

    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
    except Exception as exc:
        raise PolyTranscribeError(f"載入 basic-pitch 失敗：{exc}") from exc

    onset_threshold, frame_threshold, minimum_note_length = _default_thresholds(min_note_ms)
    min_freq = _midi_to_hz(min_pitch) if min_pitch is not None else None
    max_freq = _midi_to_hz(max_pitch) if max_pitch is not None else None

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, np.asarray(y, dtype=np.float32), sr)

        if progress:
            progress(0.2, "辨識複音音符中")
        _model_output, _midi_data, note_events = predict(
            audio_path=tmp_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )
    except Exception as exc:
        raise PolyTranscribeError(f"複音轉譜失敗：{exc}") from exc
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    events: List[NoteEvent] = []
    for note in note_events:
        # basic-pitch note_events 元素格式：(start_time, end_time, pitch_midi, amplitude, pitch_bend?)
        start_time, end_time, pitch_midi, amplitude = note[0], note[1], note[2], note[3]
        if end_time <= start_time:
            end_time = start_time + 0.05
        pitch = int(pitch_midi)
        if min_pitch is not None and pitch < min_pitch:
            continue
        if max_pitch is not None and pitch > max_pitch:
            continue
        velocity = int(np.clip(float(amplitude) * 127, 1, 127))
        events.append(
            NoteEvent(
                onset=float(start_time),
                offset=float(end_time),
                pitch=pitch,
                velocity=velocity,
                confidence=float(np.clip(float(amplitude), 0.0, 1.0)),
            )
        )

    events.sort(key=lambda e: e.onset)
    if progress:
        progress(1.0, "複音轉譜完成")
    return events


# ---------------------------------------------------------------------------
# 泛音（諧波）誤判抑制：basic-pitch 常把明顯泛音當成獨立音符報出來
# ---------------------------------------------------------------------------

# 半音差對應的泛音級數：12=第2泛音(高八度)、19=第3泛音(八度+完全五度)、
# 24=第4泛音(兩個八度)、28=第5泛音(兩個八度+大三度)。
_HARMONIC_SEMITONE_INTERVALS = (12, 19, 24, 28)
_HARMONIC_ONSET_TOL_S = 0.04
_HARMONIC_CONFIDENCE_RATIO = 0.5


def suppress_harmonics(
    events: List[NoteEvent],
    onset_tol_s: float = _HARMONIC_ONSET_TOL_S,
    confidence_ratio: float = _HARMONIC_CONFIDENCE_RATIO,
    intervals: tuple = _HARMONIC_SEMITONE_INTERVALS,
) -> List[NoteEvent]:
    """保守的泛音抑制後處理：找出「疑似是另一個音符泛音」的音符並刪除。

    規則（刻意保守，寧可漏抓也不要誤刪真正的和聲/和弦音）：音符 B 若跟某個
    音符 A 的 onset 差 < `onset_tol_s`、pitch 差剛好是 `intervals` 中的一個
    半音數（即八度、八度+五度、兩個八度、兩個八度+大三度，對應自然泛音列的
    第 2/3/4/5 泛音），且 B 的 confidence 明顯低於 A（< `confidence_ratio` 倍），
    就視為 A 的泛音誤判，刪掉 B。判斷完全基於輸入時的原始 confidence（不會
    因為前面刪過誰而連鎖影響後面的判斷），避免對同一組音符過度刪減。
    """
    if not events:
        return []

    n = len(events)
    to_remove = set()
    for i in range(n):
        a = events[i]
        for j in range(n):
            if i == j:
                continue
            b = events[j]
            if abs(b.onset - a.onset) >= onset_tol_s:
                continue
            if (b.pitch - a.pitch) not in intervals:
                continue
            if b.confidence < confidence_ratio * a.confidence:
                to_remove.add(j)

    return [e for idx, e in enumerate(events) if idx not in to_remove]
