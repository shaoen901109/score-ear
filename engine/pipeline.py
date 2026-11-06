# -*- coding: utf-8 -*-
"""轉譜主流程：對外唯一入口 transcribe()。"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from notation.quantize import quantize, remove_short_notes
from notation.score_model import (
    NoteEvent,
    Part,
    Score,
    TimeSignature,
    TranscribeOptions,
)

from . import audio_io
from .key import detect_key
from .rhythm import detect_tempo_and_beats

ProgressFn = Optional[Callable[[float, str], None]]

_TIME_SIGNATURES = {
    "4/4": TimeSignature(beats=4, beat_type=4),
    "3/4": TimeSignature(beats=3, beat_type=4),
    "6/8": TimeSignature(beats=6, beat_type=8),
}

_LOW_CONFIDENCE_THRESHOLD = 0.3

# other 與 vocals 重複旋律去重的判定門檻（見 _dedup_other_vs_vocals）。
_DEDUP_ONSET_TOL_S = 0.06
_DEDUP_OCTAVE_SEMITONES = (0, 12)

# 各聲部的預設中繼資料（id/name/instrument/clef/midi_program），交給 notation.quantize.quantize
# 的 part_meta 參數使用，由它負責組出完整的 Part。
_PART_META = {
    "piano": dict(id="P1", name="Piano", instrument="piano", clef="grand", midi_program=0),
    "vocal": dict(id="P1", name="Vocal", instrument="vocal", clef="treble", midi_program=53),
    "guitar": dict(id="P1", name="Guitar", instrument="guitar", clef="treble", midi_program=24),
    "other_solo": dict(id="P1", name="Other", instrument="other", clef="treble", midi_program=0),
    "vocals": dict(id="P1", name="Vocal", instrument="vocal", clef="treble", midi_program=53),
    "drums": dict(id="P2", name="Drums", instrument="drums", clef="percussion", midi_program=0),
    "bass": dict(id="P3", name="Bass", instrument="bass", clef="bass", midi_program=33),
    "other": dict(id="P4", name="Other", instrument="other", clef="treble", midi_program=0),
}


class TranscribeError(RuntimeError):
    """轉譜流程失敗（附中文說明），統一包裝底層例外。"""


def _report(progress: ProgressFn, fraction: float, message: str) -> None:
    if progress:
        progress(min(max(fraction, 0.0), 1.0), message)


def _resolve_time_signature(
    options: TranscribeOptions, y: np.ndarray, sr: int, beat_times: np.ndarray
) -> TimeSignature:
    ts = getattr(options, "time_signature", "auto")
    if ts == "auto":
        from .rhythm import detect_time_signature

        # 簡易自動偵測：只分辨 3/4 vs 4/4，6/8 暫不自動偵測（見 engine/rhythm.py
        # detect_time_signature 的說明），偵測失敗或資訊不足一律回退 4/4。
        try:
            detected = detect_time_signature(y, sr, beat_times)
        except Exception:
            detected = "4/4"
        return _TIME_SIGNATURES.get(detected, TimeSignature(beats=4, beat_type=4))
    return _TIME_SIGNATURES.get(ts, TimeSignature(beats=4, beat_type=4))


def _effective_use_gpu(options: TranscribeOptions) -> bool:
    if not bool(getattr(options, "use_gpu", True)):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _maybe_suppress_harmonics(events: List[NoteEvent], options: TranscribeOptions) -> List[NoteEvent]:
    """basic-pitch（transcribe_poly）結果的保守泛音抑制，依 TranscribeOptions.suppress_harmonics 開關。"""
    if not events or not bool(getattr(options, "suppress_harmonics", True)):
        return events
    from .transcribe_poly import suppress_harmonics

    return suppress_harmonics(events)


def _transcribe_solo(
    y: np.ndarray, sr: int, options: TranscribeOptions, progress: ProgressFn, use_gpu: bool
) -> Tuple[List[NoteEvent], dict]:
    hint = getattr(options, "instrument_hint", "auto")
    min_note_ms = getattr(options, "min_note_ms", 60)

    if hint == "piano":
        from .transcribe_piano import transcribe_piano

        _report(progress, 0.3, "辨識鋼琴音符")
        events = transcribe_piano(y, sr, use_gpu=use_gpu)
        meta = _PART_META["piano"]
    elif hint == "vocal":
        from .transcribe_mono import monophonize, transcribe_pyin
        from .transcribe_poly import transcribe_poly

        _report(progress, 0.3, "辨識人聲旋律")
        try:
            events = transcribe_poly(y, sr, min_note_ms=min_note_ms)
            events = _maybe_suppress_harmonics(events, options)
            events = monophonize(events)
        except Exception:
            events = transcribe_pyin(y, sr, min_note_ms=min_note_ms)
        meta = _PART_META["vocal"]
    elif hint == "guitar":
        from .transcribe_poly import transcribe_poly

        _report(progress, 0.3, "辨識吉他音符")
        events = transcribe_poly(y, sr, min_note_ms=min_note_ms)
        events = _maybe_suppress_harmonics(events, options)
        meta = _PART_META["guitar"]
    else:
        from .transcribe_poly import transcribe_poly

        _report(progress, 0.3, "辨識音符 (basic-pitch)")
        events = transcribe_poly(y, sr, min_note_ms=min_note_ms)
        events = _maybe_suppress_harmonics(events, options)
        meta = _PART_META["other_solo"]

    return events, meta


def _transcribe_multi(
    y: np.ndarray, sr: int, options: TranscribeOptions, progress: ProgressFn, use_gpu: bool
) -> Dict[str, Tuple[List[NoteEvent], dict]]:
    from .separate import separate
    from .transcribe_drums import transcribe_drums
    from .transcribe_mono import monophonize, transcribe_pyin
    from .transcribe_poly import transcribe_poly

    min_note_ms = getattr(options, "min_note_ms", 60)

    _report(progress, 0.15, "分離音源中（vocals/drums/bass/other）")
    stems = separate(y, sr, use_gpu=use_gpu)

    parts: Dict[str, Tuple[List[NoteEvent], dict]] = {}

    vocals = stems.get("vocals")
    if vocals is not None and len(vocals) > 0:
        _report(progress, 0.55, "辨識人聲旋律")
        try:
            v_events = transcribe_poly(vocals, sr, min_note_ms=min_note_ms)
            v_events = _maybe_suppress_harmonics(v_events, options)
            v_events = monophonize(v_events)
        except Exception:
            v_events = transcribe_pyin(vocals, sr, min_note_ms=min_note_ms)
        if v_events:
            parts["vocals"] = (v_events, _PART_META["vocals"])

    drums = stems.get("drums")
    if drums is not None and len(drums) > 0:
        _report(progress, 0.65, "辨識鼓聲")
        d_events = transcribe_drums(drums, sr, min_note_ms=min_note_ms)
        if d_events:
            parts["drums"] = (d_events, _PART_META["drums"])

    bass = stems.get("bass")
    if bass is not None and len(bass) > 0:
        _report(progress, 0.75, "辨識貝斯音符")
        b_events = transcribe_poly(bass, sr, min_note_ms=min_note_ms, min_pitch=28, max_pitch=60)
        b_events = _maybe_suppress_harmonics(b_events, options)
        if b_events:
            parts["bass"] = (b_events, _PART_META["bass"])

    other = stems.get("other")
    if other is not None and len(other) > 0:
        _report(progress, 0.85, "辨識其他樂器音符")
        o_events = transcribe_poly(other, sr, min_note_ms=min_note_ms)
        o_events = _maybe_suppress_harmonics(o_events, options)
        if o_events:
            parts["other"] = (o_events, _PART_META["other"])

    return parts


def _dedup_other_vs_vocals(
    other_events: List[NoteEvent], vocal_events: List[NoteEvent]
) -> Tuple[List[NoteEvent], int]:
    """去除 other 聲部裡跟 vocals 重複的旋律音符。

    demucs 分離人工混音（或音色特殊的錄音）時，常把主旋律同時漏進 vocals 與
    other 兩個 stem，導致同一段旋律在樂譜上被寫兩次。這裡用「pitch 相同或
    差八度、onset 差 < 60ms、other 的 confidence 較低」判斷 other 裡的音符是
    vocals 同一個音符的重複，符合就從 other 移除；只單向處理（不動 vocals），
    因為 vocals part 的辨識管線做過單音化，本來就比較可信。
    """
    if not other_events or not vocal_events:
        return other_events, 0

    kept: List[NoteEvent] = []
    removed = 0
    for oe in other_events:
        is_duplicate = False
        for ve in vocal_events:
            if abs(oe.pitch - ve.pitch) not in _DEDUP_OCTAVE_SEMITONES:
                continue
            if abs(oe.onset - ve.onset) >= _DEDUP_ONSET_TOL_S:
                continue
            if oe.confidence < ve.confidence:
                is_duplicate = True
                break
        if is_duplicate:
            removed += 1
        else:
            kept.append(oe)
    return kept, removed


def transcribe(path: str, options: TranscribeOptions, progress: ProgressFn = None) -> Score:
    """辨識音訊檔案，回傳完整的 Score。對外唯一入口。

    - solo 模式：不分離，依 instrument_hint 選轉譜器。
    - multi 模式：demucs 分離後，各聲部各自轉譜；全空的聲部不建 Part。
    - 任何底層例外都會包成帶中文訊息的 TranscribeError。
    """
    warnings: List[str] = []
    try:
        _report(progress, 0.0, "讀取音訊檔")
        y, sr = audio_io.load_audio(path)

        _report(progress, 0.1, "偵測速度與拍點")
        bpm_override = getattr(options, "bpm_override", None)
        bpm, beat_times, confidence = detect_tempo_and_beats(y, sr)
        if bpm_override:
            bpm = float(bpm_override)
        elif confidence < _LOW_CONFIDENCE_THRESHOLD:
            warnings.append("速度偵測信心低，建議手動確認或覆寫 BPM")

        time_signature = _resolve_time_signature(options, y, sr, beat_times)
        use_gpu = _effective_use_gpu(options)
        min_note_ms = getattr(options, "min_note_ms", 60)

        mode = getattr(options, "mode", "solo")
        if mode == "multi":
            stem_results = _transcribe_multi(y, sr, options, progress, use_gpu)
            if not stem_results:
                stem_results = {"other": ([], _PART_META["other"])}
                warnings.append("未偵測到任何樂器音源，輸出空白聲部")
            elif "other" in stem_results and "vocals" in stem_results:
                other_events, other_meta = stem_results["other"]
                vocal_events, _vocal_meta = stem_results["vocals"]
                deduped_other, removed_count = _dedup_other_vs_vocals(other_events, vocal_events)
                if removed_count > 0:
                    stem_results["other"] = (deduped_other, other_meta)
                    warnings.append(
                        f"other 聲部移除 {removed_count} 個疑似與 vocals 重複的旋律音符"
                    )
        else:
            events, meta = _transcribe_solo(y, sr, options, progress, use_gpu)
            stem_results = {meta["instrument"]: (events, meta)}

        # 統一濾掉太短的音符（TranscribeOptions.min_note_ms 真正生效的地方）。
        #
        # 各轉譜器各自有自己的「最短音符」概念（basic-pitch 的
        # minimum_note_length、pyin 內部的 min_note_ms 過濾、鼓的視窗長度），
        # 但鋼琴轉譜（transcribe_piano）完全沒有這道過濾——實測對合成鋼琴音檔
        # 跑 piano_transcription_inference，除了正確音符外還會夾雜幾毫秒長的
        # 雜訊音符（疑似泛音或轉譜模型的瞬態誤判），沒有任何地方把它們濾掉。
        # 這裡在量化之前對「所有」聲部統一套用 `remove_short_notes()`
        # （notation/quantize.py 裡原本就有這個函式，但過去沒有任何呼叫端用到
        # 它），當作最後一道防線，跟各轉譜器自己的過濾疊加也無害。
        stem_results = {
            key: (remove_short_notes(events, min_note_ms), meta)
            for key, (events, meta) in stem_results.items()
        }

        all_events: List[NoteEvent] = []
        for events, _meta in stem_results.values():
            all_events.extend(events)

        _report(progress, 0.9, "分析調性")
        key = detect_key(all_events)

        _report(progress, 0.92, "量化音符、建立樂譜")
        grid = getattr(options, "quantize", "1/16")
        # notation.quantize.quantize 預期 beat_times 是 list（會用 `if not beat_times` 判斷是否為
        # 空），numpy 陣列在元素數 > 1 時做布林判斷會丟例外，這裡轉成 list（或空陣列時給 None）。
        beat_times_list = beat_times.tolist() if beat_times is not None and len(beat_times) > 0 else None
        parts: List[Part] = []
        for events, meta in stem_results.values():
            part = quantize(events, bpm, beat_times_list, time_signature, grid, meta)
            parts.append(part)

        score = Score(
            title="",
            bpm=bpm,
            time_signature=time_signature,
            key=key,
            parts=parts,
            source_file=path,
            warnings=warnings,
        )

        _report(progress, 1.0, "完成")
        return score
    except TranscribeError:
        raise
    except Exception as exc:
        raise TranscribeError(f"轉譜失敗：{exc}") from exc
