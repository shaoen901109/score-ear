# -*- coding: utf-8 -*-
"""notation.midi 測試：輸出的 bytes 能被 mido 讀回，且音符數正確。"""

import io
import unittest

import mido

from notation.midi import score_to_midi_bytes
from notation.quantize import quantize
from notation.score_model import KeySignature, Measure, Note, NoteEvent, Part, Score, TimeSignature

TS_4_4 = TimeSignature(beats=4, beat_type=4)


def _count_note_on(mid: mido.MidiFile) -> int:
    count = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                count += 1
    return count


class TestMidiBasics(unittest.TestCase):
    def test_roundtrip_readable(self):
        events = [
            NoteEvent(onset=0.0, offset=0.5, pitch=60),
            NoteEvent(onset=0.5, offset=1.0, pitch=62),
            NoteEvent(onset=1.0, offset=1.5, pitch=64),
        ]
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="T", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        data = score_to_midi_bytes(score)
        self.assertIsInstance(data, (bytes, bytearray))
        mid = mido.MidiFile(file=io.BytesIO(data))
        self.assertEqual(_count_note_on(mid), 3)

    def test_chord_counts_as_multiple_note_on(self):
        events = [
            NoteEvent(onset=0.0, offset=1.0, pitch=60),
            NoteEvent(onset=0.0, offset=1.0, pitch=64),
            NoteEvent(onset=0.0, offset=1.0, pitch=67),
        ]
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Chord", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        data = score_to_midi_bytes(score)
        mid = mido.MidiFile(file=io.BytesIO(data))
        self.assertEqual(_count_note_on(mid), 3)

    def test_tie_merges_into_single_note_on(self):
        # 一個跨小節的長音，quantize 會拆成兩個 tie 起訖的 Note，MIDI 應合併成一個音符。
        events = [NoteEvent(onset=0.0, offset=2.5, pitch=60)]
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Tie", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        data = score_to_midi_bytes(score)
        mid = mido.MidiFile(file=io.BytesIO(data))
        self.assertEqual(_count_note_on(mid), 1)

    def test_drums_use_channel_10(self):
        score = Score(
            title="Drums",
            bpm=120.0,
            time_signature=TS_4_4,
            key=KeySignature(fifths=0, mode="major", tonic="C"),
            parts=[
                Part(
                    id="P1", name="Drums", instrument="drums", clef="percussion", midi_program=0,
                    measures=[Measure(number=1, notes=[Note(start_tick=0, duration_tick=1920, pitches=[36])])],
                )
            ],
        )
        data = score_to_midi_bytes(score)
        mid = mido.MidiFile(file=io.BytesIO(data))
        channels = {
            msg.channel for track in mid.tracks for msg in track if msg.type == "note_on"
        }
        self.assertEqual(channels, {9})

    def test_chinese_title_and_part_name_do_not_crash(self):
        # 迴歸測試：mido 預設用 latin1 編碼文字事件，中文標題/聲部名會直接丟
        # UnicodeEncodeError；notation.midi 需指定 charset="utf-8" 避免這個問題。
        events = [NoteEvent(onset=0.0, offset=0.5, pitch=60)]
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "鋼琴", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="示範樂曲", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        data = score_to_midi_bytes(score)
        mid = mido.MidiFile(file=io.BytesIO(data))
        self.assertEqual(_count_note_on(mid), 1)

    def test_empty_score_still_produces_valid_midi(self):
        part = quantize([], 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Empty", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        data = score_to_midi_bytes(score)
        mid = mido.MidiFile(file=io.BytesIO(data))
        self.assertEqual(_count_note_on(mid), 0)


if __name__ == "__main__":
    unittest.main()
