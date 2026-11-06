# -*- coding: utf-8 -*-
"""notation.musicxml 測試：用 xml.etree 解析輸出，檢查結構與時值總和。"""

import unittest
import xml.etree.ElementTree as ET

from notation.musicxml import score_to_musicxml
from notation.quantize import quantize
from notation.score_model import KeySignature, Measure, Note, NoteEvent, Part, Score, TimeSignature

TS_4_4 = TimeSignature(beats=4, beat_type=4)


def _simple_score() -> Score:
    events = [
        NoteEvent(onset=0.0, offset=0.5, pitch=60),
        NoteEvent(onset=0.5, offset=1.0, pitch=62),
        NoteEvent(onset=1.0, offset=1.5, pitch=64),
        NoteEvent(onset=1.5, offset=2.0, pitch=65),
    ]
    part = quantize(events, 120.0, None, TS_4_4, "1/16", {
        "id": "P1", "name": "Piano", "instrument": "piano", "clef": "treble", "midi_program": 0,
    })
    return Score(
        title="Test",
        bpm=120.0,
        time_signature=TS_4_4,
        key=KeySignature(fifths=0, mode="major", tonic="C"),
        parts=[part],
    )


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1]


class TestMusicXmlStructure(unittest.TestCase):
    def setUp(self):
        self.score = _simple_score()
        self.xml_str = score_to_musicxml(self.score)
        # 拿掉 DOCTYPE，讓 ET 可以直接 parse（ET 不處理外部 DTD）。
        body = self.xml_str.split("?>", 1)[1]
        if "<!DOCTYPE" in body:
            body = body.split(">", 1)[1]
        self.root = ET.fromstring(body)

    def test_is_score_partwise(self):
        self.assertEqual(_strip_ns(self.root.tag), "score-partwise")

    def test_part_count(self):
        parts = self.root.findall("part")
        self.assertEqual(len(parts), 1)

    def test_measure_count(self):
        part = self.root.find("part")
        measures = part.findall("measure")
        self.assertEqual(len(measures), 1)

    def test_measure_duration_sum_equals_divisions_times_4(self):
        part = self.root.find("part")
        for measure in part.findall("measure"):
            total = 0
            for note in measure.findall("note"):
                if note.find("chord") is not None:
                    continue  # 和弦內的音不重複計算時值
                dur = note.find("duration")
                total += int(dur.text)
            self.assertEqual(total, 480 * 4)

    def test_key_time_clef_present(self):
        attributes = self.root.find("part/measure/attributes")
        self.assertIsNotNone(attributes.find("divisions"))
        self.assertIsNotNone(attributes.find("key/fifths"))
        self.assertIsNotNone(attributes.find("time/beats"))
        self.assertIsNotNone(attributes.find("clef/sign"))

    def test_rest_has_rest_tag(self):
        events = []
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Empty", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        xml_str = score_to_musicxml(score)
        body = xml_str.split("?>", 1)[1].split(">", 1)[1]
        root = ET.fromstring(body)
        note = root.find("part/measure/note")
        self.assertIsNotNone(note.find("rest"))


class TestGrandClefTwoStaves(unittest.TestCase):
    def test_staves_and_backup(self):
        events = [
            NoteEvent(onset=0.0, offset=2.0, pitch=72),
            NoteEvent(onset=0.0, offset=2.0, pitch=48),
        ]
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "grand", "midi_program": 0,
        })
        score = Score(title="Grand", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        xml_str = score_to_musicxml(score)
        body = xml_str.split("?>", 1)[1].split(">", 1)[1]
        root = ET.fromstring(body)
        attributes = root.find("part/measure/attributes")
        self.assertIsNotNone(attributes.find("staves"))
        self.assertEqual(attributes.find("staves").text, "2")
        measure = root.find("part/measure")
        self.assertIsNotNone(measure.find("backup"))


class TestPercussionUnpitched(unittest.TestCase):
    def test_unpitched_used(self):
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
        xml_str = score_to_musicxml(score)
        body = xml_str.split("?>", 1)[1].split(">", 1)[1]
        root = ET.fromstring(body)
        note = root.find("part/measure/note")
        self.assertIsNotNone(note.find("unpitched"))


if __name__ == "__main__":
    unittest.main()
