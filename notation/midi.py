# -*- coding: utf-8 -*-
"""Score -> 標準 MIDI 檔案 bytes（用 mido）。

- 一個 Part 對應一個 track。
- 鼓（clef == "percussion"）固定走 channel 10（mido/實體上是 index 9）。
- 其餘 Part 依序分配 channel（跳過 9），超過 15 軌時循環使用（同 channel 多軌很常見，
  只要 program 不同即可分辨音色）。
- 同一 pitch 被 tie_start/tie_stop 串起的多個 Note，合併成一個長 note_on/note_off。
"""

from __future__ import annotations

import io

import mido

from .score_model import Score

DEFAULT_VELOCITY = 96


def _ticks_per_measure(score: Score) -> int:
    ts = score.time_signature
    return int(ts.beats * score.ticks_per_quarter * 4 / ts.beat_type)


def _collect_merged_notes(part, ticks_per_measure: int) -> list:
    """把一個 Part 的所有 Measure/Note 轉成 (abs_start, abs_end, pitch) 的清單，
    並把 tie_start/tie_stop 串起的片段合併成一個音符。
    """
    open_notes: dict = {}
    finalized: list = []

    for measure in sorted(part.measures, key=lambda m: m.number):
        base = (measure.number - 1) * ticks_per_measure
        for note in measure.notes:
            if not note.pitches:
                continue
            abs_start = base + note.start_tick
            abs_end = abs_start + note.duration_tick
            for pitch in note.pitches:
                key = (pitch, note.voice)
                if note.tie_stop and key in open_notes:
                    open_notes[key]["end"] = abs_end
                else:
                    if key in open_notes:
                        finalized.append(open_notes.pop(key))
                    open_notes[key] = {"start": abs_start, "end": abs_end, "pitch": pitch}
                if not note.tie_start:
                    finalized.append(open_notes.pop(key))

    finalized.extend(open_notes.values())
    finalized.sort(key=lambda n: (n["start"], n["pitch"]))
    return finalized


def score_to_midi_bytes(score: Score) -> bytes:
    # charset="utf-8"：track name 常常是中文（曲名、聲部名），mido 預設用 latin1
    # 編碼會直接丟 UnicodeEncodeError，改用 utf-8 才能安全寫入中文文字事件。
    mid = mido.MidiFile(type=1, ticks_per_beat=int(score.ticks_per_quarter), charset="utf-8")

    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(mido.MetaMessage("track_name", name=score.title or "聽歌寫譜", time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(max(1.0, score.bpm)), time=0))
    ts = score.time_signature
    tempo_track.append(
        mido.MetaMessage("time_signature", numerator=ts.beats, denominator=ts.beat_type, time=0)
    )

    ticks_per_measure = _ticks_per_measure(score)

    next_channel = 0

    def _alloc_channel() -> int:
        nonlocal next_channel
        ch = next_channel % 16
        if ch == 9:
            next_channel += 1
            ch = next_channel % 16
        next_channel += 1
        return ch

    for part in score.parts or []:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=part.name, time=0))

        is_drums = part.clef == "percussion"
        channel = 9 if is_drums else _alloc_channel()
        if not is_drums:
            track.append(
                mido.Message("program_change", program=int(part.midi_program) % 128, channel=channel, time=0)
            )

        merged = _collect_merged_notes(part, ticks_per_measure)

        events = []
        for n in merged:
            events.append((n["start"], 1, n["pitch"]))   # note_on 排在同 tick 的 note_off 之後
            events.append((n["end"], 0, n["pitch"]))      # note_off
        events.sort(key=lambda e: (e[0], e[1]))

        cursor = 0
        for abs_tick, is_on, pitch in events:
            delta = max(0, abs_tick - cursor)
            cursor = abs_tick
            if is_on:
                track.append(
                    mido.Message(
                        "note_on", note=int(pitch), velocity=DEFAULT_VELOCITY, channel=channel, time=delta
                    )
                )
            else:
                track.append(
                    mido.Message("note_off", note=int(pitch), velocity=0, channel=channel, time=delta)
                )
        track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()
