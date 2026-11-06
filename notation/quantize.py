# -*- coding: utf-8 -*-
"""把辨識階段的 NoteEvent（單位：秒）量化成拍格（tick）的 Part。

主要入口：`quantize(events, bpm, beat_times, time_signature, grid, part_meta) -> Part`

流程：
1. 用 beat_times（沒有就用等速 bpm）把每個 NoteEvent 的 onset/offset 秒數換算成
   以四分音符為單位的拍位置，再乘上 ticks_per_quarter 得到 tick。
2. onset/offset 各自取最接近的 grid 格；量化後長度不足一格的補到一格。
3. 同一個 voice 內，onset tick 完全相同的音合併成和弦（pitches 多個），
   時值取其中最長者；若合併後與下一個音重疊則截短。
4. 依小節長度切開跨小節的音符，前後兩段用 tie_start / tie_stop 串起來。
5. 音符之間的空隙補休止符（pitches=[]），使每個小節、每個 voice 的 tick 總和
   剛好等於小節長度。
6. clef=="grand" 時，pitch<60 的音走 voice=2（低音譜），其餘走 voice=1
   （高音譜），兩個 voice 各自獨立補休止；其餘 clef（含 percussion）只有
   voice=1，鼓不做和弦合併以外的特殊處理（也就是跟其他 clef 走同一條路徑）。
"""

from __future__ import annotations

import bisect
import math
from typing import Callable, Optional

from .score_model import Measure, Note, NoteEvent, Part, TimeSignature

TICKS_PER_QUARTER = 480


# ---------------------------------------------------------------------------
# 前處理：remove_short_notes / merge_short_gaps（在「秒」的世界操作，量化之前用）
# ---------------------------------------------------------------------------

def remove_short_notes(events: list, min_ms: int) -> list:
    """濾掉時長小於 min_ms 毫秒的音符事件。"""
    min_s = min_ms / 1000.0
    return [e for e in events if (e.offset - e.onset) >= min_s]


def merge_short_gaps(events: list, max_gap_ms: float = 40.0) -> list:
    """同一音高、間隔小於 max_gap_ms 的相鄰音符事件合併成一個長音符。

    常見情境：轉譜模型把一個長音誤判成中間有極短空隙的兩三個音符，
    合併後才不會在簡譜/五線譜上被切成一堆極短的音。
    """
    if not events:
        return []
    max_gap_s = max_gap_ms / 1000.0
    by_pitch: dict = {}
    for e in events:
        by_pitch.setdefault(e.pitch, []).append(e)

    merged: list = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda e: e.onset)
        current = group[0]
        for nxt in group[1:]:
            gap = nxt.onset - current.offset
            if gap <= max_gap_s and gap >= 0:
                current = NoteEvent(
                    onset=current.onset,
                    offset=max(current.offset, nxt.offset),
                    pitch=pitch,
                    velocity=max(current.velocity, nxt.velocity),
                    confidence=min(current.confidence, nxt.confidence),
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    merged.sort(key=lambda e: e.onset)
    return merged


# ---------------------------------------------------------------------------
# 秒 -> 拍位置（以四分音符為一拍）
# ---------------------------------------------------------------------------

def _seconds_to_beat_position(t: float, bpm: float, beat_times: Optional[list]) -> float:
    if not beat_times:
        beat_dur = 60.0 / bpm if bpm and bpm > 0 else 0.5
        return t / beat_dur

    n = len(beat_times)
    if n == 1:
        beat_dur = 60.0 / bpm if bpm and bpm > 0 else 0.5
        return (t - beat_times[0]) / beat_dur + 0.0

    if t <= beat_times[0]:
        seg = beat_times[1] - beat_times[0]
        return (t - beat_times[0]) / seg if seg > 0 else 0.0
    if t >= beat_times[-1]:
        seg = beat_times[-1] - beat_times[-2]
        base = n - 1
        return base + ((t - beat_times[-1]) / seg if seg > 0 else 0.0)

    idx = bisect.bisect_right(beat_times, t) - 1
    idx = max(0, min(idx, n - 2))
    seg = beat_times[idx + 1] - beat_times[idx]
    frac = (t - beat_times[idx]) / seg if seg > 0 else 0.0
    return idx + frac


def _grid_ticks(grid: str) -> int:
    """把 "1/16" 這類字串換算成 tick 數（480 分音符制下）。"""
    try:
        denom = int(str(grid).split("/")[-1])
    except (ValueError, IndexError):
        denom = 16
    ticks = (TICKS_PER_QUARTER * 4) // denom
    return max(1, ticks)


def _round_to_grid(tick: float, grid_ticks: int) -> int:
    return int(round(tick / grid_ticks)) * grid_ticks


def _ticks_per_measure(time_signature: TimeSignature) -> int:
    beats = time_signature.beats
    beat_type = time_signature.beat_type
    return int(beats * TICKS_PER_QUARTER * 4 / beat_type)


# ---------------------------------------------------------------------------
# 和弦合併 + 消除重疊
# ---------------------------------------------------------------------------

def _merge_chords(tick_events: list) -> list:
    """tick_events: list[(onset_tick, offset_tick, pitch)] -> list[[onset, duration, pitches]]"""
    if not tick_events:
        return []
    tick_events = sorted(tick_events, key=lambda e: (e[0], e[2]))
    chords: list = []
    cur_onset = tick_events[0][0]
    cur_end = tick_events[0][1]
    cur_pitches = [tick_events[0][2]]
    for onset, offset, pitch in tick_events[1:]:
        if onset == cur_onset:
            cur_end = max(cur_end, offset)
            if pitch not in cur_pitches:
                cur_pitches.append(pitch)
        else:
            chords.append([cur_onset, cur_end - cur_onset, sorted(cur_pitches)])
            cur_onset = onset
            cur_end = offset
            cur_pitches = [pitch]
    chords.append([cur_onset, cur_end - cur_onset, sorted(cur_pitches)])
    return chords


def _trim_overlaps(chords: list, grid_ticks: int) -> list:
    for i in range(len(chords) - 1):
        cur = chords[i]
        nxt = chords[i + 1]
        cur_end = cur[0] + cur[1]
        if cur_end > nxt[0]:
            new_dur = max(grid_ticks, nxt[0] - cur[0])
            cur[1] = new_dur
    return chords


# ---------------------------------------------------------------------------
# 把一個 voice 的和弦序列切成各小節的 Note（含補休止、跨小節 tie）
# ---------------------------------------------------------------------------

def _place_voice(chords: list, ticks_per_measure: int, voice: int, full_length: int) -> dict:
    measures: dict = {}

    def _add(measure_no: int, note: Note) -> None:
        measures.setdefault(measure_no, []).append(note)

    pos = 0
    idx = 0
    n = len(chords)
    while pos < full_length:
        if idx < n and chords[idx][0] == pos:
            onset, dur, pitches = chords[idx]
            idx += 1
            remaining = dur
            seg_start = pos
            first = True
            while remaining > 0:
                seg_measure_no = seg_start // ticks_per_measure + 1
                seg_measure_end = seg_measure_no * ticks_per_measure
                seg_len = min(remaining, seg_measure_end - seg_start)
                is_last = (seg_len == remaining)
                tie_start = not is_last
                tie_stop = not first
                local_start = seg_start - (seg_measure_no - 1) * ticks_per_measure
                _add(
                    seg_measure_no,
                    Note(
                        start_tick=local_start,
                        duration_tick=seg_len,
                        pitches=list(pitches),
                        tie_start=tie_start,
                        tie_stop=tie_stop,
                        voice=voice,
                    ),
                )
                seg_start += seg_len
                remaining -= seg_len
                first = False
            pos = seg_start
        else:
            next_onset = chords[idx][0] if idx < n else full_length
            measure_no = pos // ticks_per_measure + 1
            measure_end = measure_no * ticks_per_measure
            gap_end = min(next_onset, measure_end)
            gap_len = gap_end - pos
            if gap_len > 0:
                local_start = pos - (measure_no - 1) * ticks_per_measure
                _add(
                    measure_no,
                    Note(start_tick=local_start, duration_tick=gap_len, pitches=[], voice=voice),
                )
                pos = gap_end
            else:
                # 理論上不會發生（gap_len<=0 代表 chords[idx][0] < pos），保底避免死圈
                pos = gap_end if gap_end > pos else pos + 1
    return measures


# ---------------------------------------------------------------------------
# 主函式
# ---------------------------------------------------------------------------

def quantize(
    events: list,
    bpm: float,
    beat_times: Optional[list],
    time_signature: TimeSignature,
    grid: str,
    part_meta: dict,
) -> Part:
    """把一組 NoteEvent（秒）量化成一個 Part（拍格 tick）。

    part_meta: dict，至少含 id/name/instrument/clef/midi_program。
    """
    ticks_per_measure = _ticks_per_measure(time_signature)
    grid_ticks = _grid_ticks(grid)
    clef = part_meta.get("clef", "treble")

    raw_tick_events = []
    for e in events or []:
        onset_raw = _seconds_to_beat_position(e.onset, bpm, beat_times) * TICKS_PER_QUARTER
        offset_raw = _seconds_to_beat_position(e.offset, bpm, beat_times) * TICKS_PER_QUARTER
        onset_q = _round_to_grid(onset_raw, grid_ticks)
        offset_q = _round_to_grid(offset_raw, grid_ticks)
        if offset_q <= onset_q:
            offset_q = onset_q + grid_ticks
        raw_tick_events.append([onset_q, offset_q, int(e.pitch)])

    # 把整個 Part 往右平移，讓最早的音符對齊 tick 0。
    #
    # 背景：`_seconds_to_beat_position` 是相對 `beat_times[0]`（第一個「偵測到」的
    # 拍點，常常不是音檔裡第一個音符發生的時間，例如節拍器抓到的第一拍落在
    # 0.8 秒，但第一個音符在 0 秒）算出來的位置，音檔開頭早於第一個偵測拍點的
    # 音符會得到負的 tick。以前的做法是逐一把負 tick clip 成 0，這會讓開頭
    # 好幾個「不同時間點」的音符全部疊在同一個 tick 0 上，破壞開頭的節奏（實測：
    # C 大調音階測試 fixture 的前三個音符全部被 clip 成同一個音）。改成用
    # 「整體平移」保留這些音符彼此的相對間隔——只要 grid 量化後最小的 tick
    # 還是負的，就把所有 tick 一起加上同一個偏移量，讓最小值變成 0。
    if raw_tick_events:
        min_onset = min(t[0] for t in raw_tick_events)
        if min_onset < 0:
            shift = -min_onset
            for t in raw_tick_events:
                t[0] += shift
                t[1] += shift

    tick_events = [(int(o), int(off), p) for o, off, p in raw_tick_events]

    if clef == "grand":
        groups = {
            1: [t for t in tick_events if t[2] >= 60],
            2: [t for t in tick_events if t[2] < 60],
        }
    else:
        groups = {1: tick_events}

    chords_by_voice = {}
    max_end = 0
    for voice, evs in groups.items():
        chords = _trim_overlaps(_merge_chords(evs), grid_ticks)
        chords_by_voice[voice] = chords
        if chords:
            last = chords[-1]
            max_end = max(max_end, last[0] + last[1])

    n_measures = max(1, math.ceil(max_end / ticks_per_measure)) if max_end > 0 else 1
    full_length = n_measures * ticks_per_measure

    measures_map: dict = {}
    for voice, chords in chords_by_voice.items():
        placed = _place_voice(chords, ticks_per_measure, voice, full_length)
        for mno, notes in placed.items():
            measures_map.setdefault(mno, []).extend(notes)

    measures = []
    for i in range(1, n_measures + 1):
        notes = sorted(measures_map.get(i, []), key=lambda nt: (nt.start_tick, nt.voice))
        measures.append(Measure(number=i, notes=notes))

    return Part(
        id=str(part_meta.get("id", "P1")),
        name=str(part_meta.get("name", "Part")),
        instrument=str(part_meta.get("instrument", "other")),
        clef=clef,
        midi_program=int(part_meta.get("midi_program", 0)),
        measures=measures,
    )
