# -*- coding: utf-8 -*-
"""樂譜共用資料模型。

全部使用 dataclass，並提供 to_dict()/from_dict() 以便在 app 層與 engine 層之間
用純 dict（可直接 json.dumps）傳遞。Score 另外提供 to_json()/from_json()。

備註（與 docs/ARCHITECTURE.md 的差異，皆為讓 dataclass 語法合法所做的調整，
語意不變，欄位名稱與型別完全保留）：
- Score：架構文件中欄位順序為
  title, bpm, time_signature, key, ticks_per_quarter=480, parts, source_file="", warnings
  但 dataclass 規則是「有預設值的欄位不能在無預設值欄位之前」，parts/warnings 在文件中
  沒有預設值卻排在有預設值欄位之後，會直接是 SyntaxError。這裡把欄位重新排序為
  必要欄位在前、有預設值欄位在後，並給 parts / warnings 補上 `field(default_factory=list)`
  的空 list 預設值（語意等同「未提供時視為空列表」），欄位名稱、型別皆未變動。
- Measure.notes / Part.measures：文件未寫預設值，這裡加上 `field(default_factory=list)`
  方便呼叫端先建立空殼再逐步 append，型別與語意不變。
- TranscribeOptions：文件只寫了型別未逐一寫預設值，這裡依文件敘述補上合理預設
  （mode 預設 "solo"，其餘依文件內文字面值：bpm_override=None、
  time_signature="auto"、min_note_ms=60、quantize="1/16"、use_gpu=True）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 小工具：泛用的 to_dict / from_dict（給下面每個 dataclass 用）
# ---------------------------------------------------------------------------

def _value_to_dict(value: Any) -> Any:
    """遞迴把 dataclass / list / 純值轉成可 JSON 序列化的結構。"""
    if is_dataclass(value):
        return value.to_dict() if hasattr(value, "to_dict") else _generic_to_dict(value)
    if isinstance(value, list):
        return [_value_to_dict(v) for v in value]
    if isinstance(value, tuple):
        return [_value_to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: _value_to_dict(v) for k, v in value.items()}
    return value


def _generic_to_dict(obj: Any) -> dict:
    return {f.name: _value_to_dict(getattr(obj, f.name)) for f in fields(obj)}


# ---------------------------------------------------------------------------
# NoteEvent：辨識階段的原始音符（單位：秒）
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    onset: float             # 起始秒
    offset: float            # 結束秒（> onset）
    pitch: int                # MIDI 音高 0-127；鼓用 GM 打擊音號
    velocity: int = 80        # 1-127
    confidence: float = 1.0   # 0-1

    def to_dict(self) -> dict:
        return _generic_to_dict(self)

    @staticmethod
    def from_dict(d: dict) -> "NoteEvent":
        return NoteEvent(
            onset=float(d["onset"]),
            offset=float(d["offset"]),
            pitch=int(d["pitch"]),
            velocity=int(d.get("velocity", 80)),
            confidence=float(d.get("confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# KeySignature / TimeSignature
# ---------------------------------------------------------------------------

@dataclass
class KeySignature:
    fifths: int    # -7..7（負為降記號）
    mode: str      # "major" | "minor"
    tonic: str     # 例 "C", "F#", "Bb"

    def to_dict(self) -> dict:
        return _generic_to_dict(self)

    @staticmethod
    def from_dict(d: dict) -> "KeySignature":
        return KeySignature(
            fifths=int(d["fifths"]),
            mode=str(d["mode"]),
            tonic=str(d["tonic"]),
        )


@dataclass
class TimeSignature:
    beats: int       # 4
    beat_type: int   # 4

    def to_dict(self) -> dict:
        return _generic_to_dict(self)

    @staticmethod
    def from_dict(d: dict) -> "TimeSignature":
        return TimeSignature(beats=int(d["beats"]), beat_type=int(d["beat_type"]))


# ---------------------------------------------------------------------------
# Note：量化後音符（單位：拍格 tick，每四分音符 = 480 tick）
# ---------------------------------------------------------------------------

@dataclass
class Note:
    start_tick: int           # 相對小節開頭
    duration_tick: int        # > 0
    pitches: list             # list[int]；空 list = 休止符；多個 = 和弦
    tie_start: bool = False   # 跨小節連結線
    tie_stop: bool = False
    voice: int = 1

    def to_dict(self) -> dict:
        return _generic_to_dict(self)

    @staticmethod
    def from_dict(d: dict) -> "Note":
        return Note(
            start_tick=int(d["start_tick"]),
            duration_tick=int(d["duration_tick"]),
            pitches=[int(p) for p in d.get("pitches", [])],
            tie_start=bool(d.get("tie_start", False)),
            tie_stop=bool(d.get("tie_stop", False)),
            voice=int(d.get("voice", 1)),
        )


@dataclass
class Measure:
    number: int                              # 從 1 開始
    notes: list = field(default_factory=list)  # list[Note]，依 start_tick 排序

    def to_dict(self) -> dict:
        return {"number": self.number, "notes": [n.to_dict() for n in self.notes]}

    @staticmethod
    def from_dict(d: dict) -> "Measure":
        return Measure(
            number=int(d["number"]),
            notes=[Note.from_dict(n) for n in d.get("notes", [])],
        )


@dataclass
class Part:
    id: str               # "P1"...
    name: str              # "Piano", "Vocal", "Bass", "Drums", "Other"
    instrument: str        # "piano" | "vocal" | "bass" | "drums" | "guitar" | "other"
    clef: str              # "treble" | "bass" | "percussion" | "grand"
    midi_program: int      # GM program 0-127；鼓忽略
    measures: list = field(default_factory=list)  # list[Measure]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "instrument": self.instrument,
            "clef": self.clef,
            "midi_program": self.midi_program,
            "measures": [m.to_dict() for m in self.measures],
        }

    @staticmethod
    def from_dict(d: dict) -> "Part":
        return Part(
            id=str(d["id"]),
            name=str(d["name"]),
            instrument=str(d["instrument"]),
            clef=str(d["clef"]),
            midi_program=int(d["midi_program"]),
            measures=[Measure.from_dict(m) for m in d.get("measures", [])],
        )


# ---------------------------------------------------------------------------
# Score：整份樂譜
# ---------------------------------------------------------------------------

@dataclass
class Score:
    title: str
    bpm: float
    time_signature: TimeSignature
    key: KeySignature
    parts: list = field(default_factory=list)     # list[Part]
    ticks_per_quarter: int = 480
    source_file: str = ""
    warnings: list = field(default_factory=list)  # list[str]，例如「速度偵測信心低」

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "bpm": self.bpm,
            "time_signature": self.time_signature.to_dict(),
            "key": self.key.to_dict(),
            "parts": [p.to_dict() for p in self.parts],
            "ticks_per_quarter": self.ticks_per_quarter,
            "source_file": self.source_file,
            "warnings": list(self.warnings),
        }

    @staticmethod
    def from_dict(d: dict) -> "Score":
        return Score(
            title=str(d.get("title", "")),
            bpm=float(d.get("bpm", 120.0)),
            time_signature=TimeSignature.from_dict(d["time_signature"]),
            key=KeySignature.from_dict(d["key"]),
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
            ticks_per_quarter=int(d.get("ticks_per_quarter", 480)),
            source_file=str(d.get("source_file", "")),
            warnings=list(d.get("warnings", [])),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "Score":
        return Score.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# TranscribeOptions：轉譜選項（供 engine.pipeline 與 app 層共用）
# ---------------------------------------------------------------------------

@dataclass
class TranscribeOptions:
    mode: str = "solo"                    # "solo" | "multi"
    instrument_hint: str = "auto"         # "auto"|"piano"|"vocal"|"guitar"|"other"
    bpm_override: Optional[float] = None
    time_signature: str = "auto"          # "auto"|"4/4"|"3/4"|"6/8"
    min_note_ms: int = 60
    quantize: str = "1/16"                # "1/16"|"1/8"|"1/32"
    use_gpu: bool = True
    suppress_harmonics: bool = True       # 對 basic-pitch 複音轉譜結果做保守的泛音抑制後處理

    def to_dict(self) -> dict:
        return _generic_to_dict(self)

    @staticmethod
    def from_dict(d: dict) -> "TranscribeOptions":
        return TranscribeOptions(
            mode=str(d.get("mode", "solo")),
            instrument_hint=str(d.get("instrument_hint", "auto")),
            bpm_override=(None if d.get("bpm_override") is None else float(d["bpm_override"])),
            time_signature=str(d.get("time_signature", "auto")),
            min_note_ms=int(d.get("min_note_ms", 60)),
            quantize=str(d.get("quantize", "1/16")),
            use_gpu=bool(d.get("use_gpu", True)),
            suppress_harmonics=bool(d.get("suppress_harmonics", True)),
        )
