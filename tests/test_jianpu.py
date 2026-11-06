# -*- coding: utf-8 -*-
"""notation.jianpu 測試：輸出是合法 XML，且含預期的數字文字。"""

import unittest
import xml.etree.ElementTree as ET

from notation.jianpu import score_to_jianpu_svg
from notation.quantize import quantize
from notation.score_model import KeySignature, Score, TimeSignature
from notation.score_model import NoteEvent

TS_4_4 = TimeSignature(beats=4, beat_type=4)


def _all_text(root) -> str:
    return "".join(el.text or "" for el in root.iter())


class TestJianpuSvg(unittest.TestCase):
    def test_valid_xml(self):
        events = [NoteEvent(onset=0.0, offset=0.5, pitch=60)]
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Test", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        svg = score_to_jianpu_svg(score)
        root = ET.fromstring(svg)  # 若不是合法 XML 會丟例外
        self.assertTrue(root.tag.endswith("svg"))

    def test_contains_expected_degree_digit(self):
        # C 大調下，MIDI 60（middle C）是 1（Do）。
        events = [NoteEvent(onset=0.0, offset=0.5, pitch=60)]
        part = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Test", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        svg = score_to_jianpu_svg(score)
        root = ET.fromstring(svg)
        text = _all_text(root)
        self.assertIn("1", text)

    def test_rest_renders_zero(self):
        part = quantize([], 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Empty", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])
        svg = score_to_jianpu_svg(score)
        root = ET.fromstring(svg)
        text = _all_text(root)
        self.assertIn("0", text)

    def test_part_index_filters_output(self):
        events = [NoteEvent(onset=0.0, offset=0.5, pitch=60)]
        part1 = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P1", "name": "PartOne", "clef": "treble", "midi_program": 0,
        })
        part2 = quantize(events, 120.0, None, TS_4_4, "1/16", {
            "id": "P2", "name": "PartTwo", "clef": "treble", "midi_program": 0,
        })
        score = Score(title="Two", bpm=120.0, time_signature=TS_4_4,
                      key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part1, part2])
        svg_all = score_to_jianpu_svg(score)
        svg_one = score_to_jianpu_svg(score, part_index=0)
        self.assertIn("PartOne", svg_all)
        self.assertIn("PartTwo", svg_all)
        self.assertIn("PartOne", svg_one)
        self.assertNotIn("PartTwo", svg_one)

    def test_drum_symbols_used(self):
        from notation.score_model import Measure, Note, Part

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
        svg = score_to_jianpu_svg(score)
        root = ET.fromstring(svg)
        text = _all_text(root)
        self.assertIn("X", text)


if __name__ == "__main__":
    unittest.main()
