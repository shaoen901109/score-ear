# -*- coding: utf-8 -*-
"""Score -> MusicXML 3.1 (partwise) 字串。

不依賴 music21，用 xml.etree.ElementTree 自己組樹再序列化，並手動補上
DOCTYPE 宣告（verovio / MuseScore 都吃得下不含 DOCTYPE 的版本，但補上較保險）。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from .score_model import Note, Part, Score

DIVISIONS = 480  # 與 Score.ticks_per_quarter 對齊（四分音符 = 480 tick）

# 時值對照表：(tick 數, type 名稱, 附點數)，由大到小，供貪婪拆解使用。
_NOTE_VALUE_TABLE = [
    (2880, "whole", 1),
    (1920, "whole", 0),
    (1440, "half", 1),
    (960, "half", 0),
    (720, "quarter", 1),
    (480, "quarter", 0),
    (360, "eighth", 1),
    (240, "eighth", 0),
    (180, "16th", 1),
    (120, "16th", 0),
    (90, "32nd", 1),
    (60, "32nd", 0),
]


def _duration_to_components(duration_tick: int) -> list:
    """把任意 tick 時值貪婪拆解成 [(type, dots, ticks), ...]。

    若時值本身就能對到單一（可能附點）音符則只回傳一個 component；
    否則拆成多個，呼叫端負責在這些 component 之間補 tie。
    """
    remaining = int(duration_tick)
    out = []
    guard = 0
    while remaining > 0 and guard < 64:
        guard += 1
        for ticks, type_name, dots in _NOTE_VALUE_TABLE:
            if ticks <= remaining:
                out.append((type_name, dots, ticks))
                remaining -= ticks
                break
        else:
            # 比最小格還小（理論上不會發生，因量化最小格已對齊），直接吃掉剩餘值。
            out.append(("32nd", 0, remaining))
            remaining = 0
    return out or [("quarter", 0, duration_tick)]


def _pitch_to_step_alter_octave(pitch: int):
    """MIDI pitch -> (step, alter, octave)，一律用升記號表示黑鍵。"""
    names = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    pc = pitch % 12
    octave = pitch // 12 - 1  # MIDI 60 = C4
    return names[pc], alters[pc], octave


def _clef_elements(clef: str):
    """回傳 [(sign, line, staff_number_or_None), ...]。"""
    if clef == "treble":
        return [("G", 2, None)]
    if clef == "bass":
        return [("F", 4, None)]
    if clef == "percussion":
        return [("percussion", None, None)]
    if clef == "grand":
        return [("G", 2, 1), ("F", 4, 2)]
    return [("G", 2, None)]


def _sub(parent, tag, text=None, **attrib):
    el = ET.SubElement(parent, tag, {k: str(v) for k, v in attrib.items()})
    if text is not None:
        el.text = str(text)
    return el


def _build_note_element(
    parent,
    pitches_or_none,
    idx_in_chord: int,
    duration: int,
    type_name: str,
    dots: int,
    voice: int,
    staff: Optional[int],
    is_rest: bool,
    is_percussion: bool,
    tie_start: bool,
    tie_stop: bool,
):
    note_el = ET.Element("note")
    if idx_in_chord > 0:
        ET.SubElement(note_el, "chord")

    if is_rest:
        ET.SubElement(note_el, "rest")
    elif is_percussion:
        pitch_val = pitches_or_none
        unpitched = ET.SubElement(note_el, "unpitched")
        _sub(unpitched, "display-step", "B")
        _sub(unpitched, "display-octave", 4)
    else:
        pitch_val = pitches_or_none
        step, alter, octave = _pitch_to_step_alter_octave(pitch_val)
        pitch_el = ET.SubElement(note_el, "pitch")
        _sub(pitch_el, "step", step)
        if alter:
            _sub(pitch_el, "alter", alter)
        _sub(pitch_el, "octave", octave)

    _sub(note_el, "duration", duration)
    if not is_rest:
        # <tie> 的順序規定是先 stop 再 start，且要緊接在 <duration> 之後。
        if tie_stop:
            _sub(note_el, "tie", type="stop")
        if tie_start:
            _sub(note_el, "tie", type="start")
    _sub(note_el, "voice", voice)
    _sub(note_el, "type", type_name)
    for _ in range(dots):
        ET.SubElement(note_el, "dot")
    if staff is not None:
        _sub(note_el, "staff", staff)

    if not is_rest and (tie_start or tie_stop):
        notations = ET.SubElement(note_el, "notations")
        if tie_stop:
            _sub(notations, "tied", type="stop")
        if tie_start:
            _sub(notations, "tied", type="start")

    parent.append(note_el)


def _emit_note(measure_el, note: Note, is_percussion: bool, needs_staff_split: bool):
    """把一個量化後的 Note（可能是和弦、可能需要拆成多個 tie 音）寫進 <measure>。"""
    components = _duration_to_components(note.duration_tick)
    k = len(components)
    pitches = note.pitches
    is_rest = len(pitches) == 0
    # 只有 grand（雙譜表）才需要標 <staff>；voice 1 -> staff 1、voice 2 -> staff 2。
    staff = note.voice if needs_staff_split else None

    for i, (type_name, dots, ticks) in enumerate(components):
        if k == 1:
            comp_tie_stop = note.tie_stop
            comp_tie_start = note.tie_start
        else:
            comp_tie_stop = note.tie_stop if i == 0 else True
            comp_tie_start = note.tie_start if i == k - 1 else True

        if is_rest:
            _build_note_element(
                measure_el, None, 0, ticks, type_name, dots, note.voice, staff,
                True, is_percussion, False, False,
            )
        else:
            for j, p in enumerate(pitches):
                _build_note_element(
                    measure_el, p, j, ticks, type_name, dots, note.voice, staff,
                    False, is_percussion, comp_tie_start, comp_tie_stop,
                )


def _part_needs_staff(part: Part) -> bool:
    return part.clef == "grand"


def score_to_musicxml(score: Score) -> str:
    root = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(root, "work")
    _sub(work, "work-title", score.title or "")

    ident = ET.SubElement(root, "identification")
    encoding = ET.SubElement(ident, "encoding")
    _sub(encoding, "software", "聽歌寫譜")

    part_list = ET.SubElement(root, "part-list")
    parts = score.parts or []
    for part in parts:
        score_part = ET.SubElement(part_list, "score-part", id=part.id)
        _sub(score_part, "part-name", part.name)
        instr = ET.SubElement(score_part, "score-instrument", id=f"{part.id}-I1")
        _sub(instr, "instrument-name", part.name)
        midi_instr = ET.SubElement(score_part, "midi-instrument", id=f"{part.id}-I1")
        if part.clef == "percussion":
            _sub(midi_instr, "midi-channel", 10)
            _sub(midi_instr, "midi-unpitched", 36)
        else:
            _sub(midi_instr, "midi-channel", 1)
            _sub(midi_instr, "midi-program", int(part.midi_program) + 1)

    if not parts:
        # 空輸入：仍需輸出一個合法、含一小節休止符的 part，避免下游解析崩潰。
        from .score_model import Measure as _Measure
        from .score_model import Note as _Note
        from .score_model import Part as _Part

        dummy = _Part(
            id="P1", name="Part 1", instrument="other", clef="treble", midi_program=0,
            measures=[_Measure(number=1, notes=[_Note(start_tick=0, duration_tick=DIVISIONS * 4, pitches=[])])],
        )
        score_part = ET.SubElement(part_list, "score-part", id=dummy.id)
        _sub(score_part, "part-name", dummy.name)
        parts = [dummy]

    n_measures = max((len(p.measures) for p in parts), default=1) or 1

    for part in parts:
        is_percussion = part.clef == "percussion"
        needs_staff_split = _part_needs_staff(part)
        part_el = ET.SubElement(root, "part", id=part.id)

        measures_by_no = {m.number: m for m in part.measures}

        for m_no in range(1, n_measures + 1):
            measure_el = ET.SubElement(part_el, "measure", number=str(m_no))
            measure = measures_by_no.get(m_no)

            if m_no == 1:
                attributes = ET.SubElement(measure_el, "attributes")
                _sub(attributes, "divisions", DIVISIONS)
                key_el = ET.SubElement(attributes, "key")
                _sub(key_el, "fifths", score.key.fifths)
                _sub(key_el, "mode", score.key.mode)
                time_el = ET.SubElement(attributes, "time")
                _sub(time_el, "beats", score.time_signature.beats)
                _sub(time_el, "beat-type", score.time_signature.beat_type)
                if needs_staff_split:
                    _sub(attributes, "staves", 2)
                clefs = _clef_elements(part.clef)
                for sign, line, staff_no in clefs:
                    clef_attrib = {"number": str(staff_no)} if staff_no is not None else {}
                    clef_el = ET.SubElement(attributes, "clef", clef_attrib)
                    _sub(clef_el, "sign", sign)
                    if line is not None:
                        _sub(clef_el, "line", line)

                direction = ET.SubElement(measure_el, "direction", placement="above")
                direction_type = ET.SubElement(direction, "direction-type")
                metronome = ET.SubElement(direction_type, "metronome")
                _sub(metronome, "beat-unit", "quarter")
                _sub(metronome, "per-minute", int(round(score.bpm)))
                _sub(direction, "sound", tempo=round(score.bpm, 2))

            if measure is None:
                # 這個 part 在此小節沒有內容（理論上不會發生，因 quantize 會補滿），
                # 補一個全小節休止符避免拍號對不上。
                ticks_per_measure = int(
                    score.time_signature.beats * DIVISIONS * 4 / score.time_signature.beat_type
                )
                filler = Note(start_tick=0, duration_tick=ticks_per_measure, pitches=[])
                _emit_note(measure_el, filler, is_percussion, needs_staff_split)
                continue

            notes_by_voice: dict = {}
            for note in measure.notes:
                notes_by_voice.setdefault(note.voice, []).append(note)

            voice_order = sorted(notes_by_voice.keys())
            for vi, voice in enumerate(voice_order):
                if vi > 0:
                    prev_voice_total = sum(n.duration_tick for n in notes_by_voice[voice_order[vi - 1]])
                    backup = ET.SubElement(measure_el, "backup")
                    _sub(backup, "duration", prev_voice_total)
                for note in sorted(notes_by_voice[voice], key=lambda n: n.start_tick):
                    _emit_note(measure_el, note, is_percussion, needs_staff_split)

    xml_str = ET.tostring(root, encoding="unicode")
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC '
        '"-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    )
    return header + xml_str
