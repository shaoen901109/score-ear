# -*- coding: utf-8 -*-
"""單音化後處理，以及 librosa.pyin 備援單音轉譜。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from notation.score_model import NoteEvent

_MERGE_GAP_S = 0.03  # 相鄰同音高音符若間隔小於此值視為可合併


def monophonize(events: List[NoteEvent], merge_gap_s: float = _MERGE_GAP_S) -> List[NoteEvent]:
    """把可能重疊的複音事件轉成單音。

    做法：依 confidence 由高到低排序，依序「佔用」時間軸；後面處理的事件若與已佔用
    區段重疊，重疊部分會被裁掉（裁到完全不重疊為止；整段都被蓋住則整個刪除）。
    最後把相鄰、音高相同、間隔很短的音符合併成一個。
    """
    if not events:
        return []

    ordered = sorted(events, key=lambda e: e.confidence, reverse=True)
    occupied: List[tuple] = []  # 已保留的 (start, end) 區間
    kept: List[NoteEvent] = []

    for ev in ordered:
        onset, offset = float(ev.onset), float(ev.offset)
        if offset <= onset:
            continue

        segments = [(onset, offset)]
        for occ_start, occ_end in occupied:
            next_segments = []
            for seg_start, seg_end in segments:
                if occ_end <= seg_start or occ_start >= seg_end:
                    next_segments.append((seg_start, seg_end))
                    continue
                if occ_start > seg_start:
                    next_segments.append((seg_start, occ_start))
                if occ_end < seg_end:
                    next_segments.append((occ_end, seg_end))
            segments = next_segments
            if not segments:
                break

        for seg_start, seg_end in segments:
            if seg_end - seg_start <= 1e-6:
                continue
            kept.append(
                NoteEvent(
                    onset=seg_start,
                    offset=seg_end,
                    pitch=ev.pitch,
                    velocity=ev.velocity,
                    confidence=ev.confidence,
                )
            )
            occupied.append((seg_start, seg_end))
        occupied.sort()

    kept.sort(key=lambda e: e.onset)
    return _merge_adjacent_same_pitch(kept, merge_gap_s)


def _merge_adjacent_same_pitch(events: List[NoteEvent], merge_gap_s: float) -> List[NoteEvent]:
    if not events:
        return []
    merged: List[NoteEvent] = [events[0]]
    for ev in events[1:]:
        last = merged[-1]
        if ev.pitch == last.pitch and (ev.onset - last.offset) <= merge_gap_s:
            merged[-1] = NoteEvent(
                onset=last.onset,
                offset=max(last.offset, ev.offset),
                pitch=last.pitch,
                velocity=max(last.velocity, ev.velocity),
                confidence=max(last.confidence, ev.confidence),
            )
        else:
            merged.append(ev)
    return merged


def transcribe_pyin(
    y: np.ndarray,
    sr: int,
    min_note_ms: int = 60,
    fmin: float = 65.0,    # 約 C2
    fmax: float = 1200.0,  # 約 D6
) -> List[NoteEvent]:
    """用 librosa.pyin 抓基頻軌，切成單音音符序列（備援用，適合人聲/單旋律樂器）。

    以每個 frame 的音高四捨五入到最近 MIDI 音高後分段，過濾掉短於 min_note_ms 的音符。
    """
    import librosa

    y = np.asarray(y, dtype=np.float32) if y is not None else np.array([], dtype=np.float32)
    if len(y) == 0:
        return []

    hop_length = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    midi = np.full_like(f0, np.nan, dtype=np.float64)
    valid = np.asarray(voiced_flag, dtype=bool) & ~np.isnan(f0)
    midi[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)
    rounded = np.round(midi)

    events: List[NoteEvent] = []
    state = {"start_idx": None, "pitch": None, "confs": []}

    def _flush(end_idx: int) -> None:
        if state["start_idx"] is None or state["pitch"] is None:
            return
        onset = float(times[state["start_idx"]])
        offset = float(times[end_idx]) if end_idx < len(times) else onset + hop_length / sr
        if (offset - onset) * 1000.0 >= min_note_ms:
            confs = state["confs"]
            conf = float(np.mean(confs)) if confs else 0.5
            events.append(
                NoteEvent(
                    onset=onset,
                    offset=offset,
                    pitch=int(state["pitch"]),
                    velocity=80,
                    confidence=float(np.clip(conf, 0.0, 1.0)),
                )
            )
        state["start_idx"] = None
        state["pitch"] = None
        state["confs"] = []

    for i in range(len(rounded)):
        if not valid[i] or np.isnan(rounded[i]):
            _flush(i)
            continue
        pitch_i = int(rounded[i])
        if state["pitch"] is None:
            state["start_idx"] = i
            state["pitch"] = pitch_i
            state["confs"] = [float(voiced_prob[i])]
        elif pitch_i == state["pitch"]:
            state["confs"].append(float(voiced_prob[i]))
        else:
            _flush(i)
            state["start_idx"] = i
            state["pitch"] = pitch_i
            state["confs"] = [float(voiced_prob[i])]
    _flush(len(rounded))

    return events
