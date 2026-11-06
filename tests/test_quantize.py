# -*- coding: utf-8 -*-
"""notation.quantize 測試：小節總長守恆、和弦合併、跨小節 tie、grand 分譜、空輸入。"""

import unittest

from notation.quantize import merge_short_gaps, quantize, remove_short_notes
from notation.score_model import NoteEvent, TimeSignature

TS_4_4 = TimeSignature(beats=4, beat_type=4)
BPM = 120.0  # 每拍 0.5 秒


def _measure_total_by_voice(measure):
    totals = {}
    for n in measure.notes:
        totals[n.voice] = totals.get(n.voice, 0) + n.duration_tick
    return totals


class TestQuantizeBasics(unittest.TestCase):
    def test_empty_input_produces_one_rest_measure(self):
        part = quantize([], BPM, None, TS_4_4, "1/16", {"id": "P1", "name": "Piano", "clef": "treble"})
        self.assertEqual(len(part.measures), 1)
        notes = part.measures[0].notes
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].pitches, [])
        self.assertEqual(notes[0].duration_tick, 1920)  # 4/4 一小節 = 1920 tick

    def test_measure_total_ticks_conserved(self):
        events = [
            NoteEvent(onset=0.0, offset=0.5, pitch=60),
            NoteEvent(onset=0.5, offset=1.0, pitch=62),
            NoteEvent(onset=1.75, offset=2.5, pitch=64),  # 跨拍，留空隙給休止符
        ]
        part = quantize(events, BPM, None, TS_4_4, "1/16", {"id": "P1", "name": "X", "clef": "treble"})
        for measure in part.measures:
            totals = _measure_total_by_voice(measure)
            for voice, total in totals.items():
                self.assertEqual(total, 1920, f"measure {measure.number} voice {voice} 總長不對")

    def test_chord_merge_same_onset(self):
        events = [
            NoteEvent(onset=0.0, offset=1.0, pitch=60),
            NoteEvent(onset=0.0, offset=1.0, pitch=64),
            NoteEvent(onset=0.0, offset=1.0, pitch=67),
        ]
        part = quantize(events, BPM, None, TS_4_4, "1/16", {"id": "P1", "name": "X", "clef": "treble"})
        chord_notes = [n for n in part.measures[0].notes if n.pitches]
        self.assertEqual(len(chord_notes), 1)
        self.assertEqual(sorted(chord_notes[0].pitches), [60, 64, 67])

    def test_tie_across_measure_boundary(self):
        # 一個從第 1 拍開始、長度 5 拍的音符（4/4 下會跨到第 2 小節）。
        events = [NoteEvent(onset=0.0, offset=2.5, pitch=60)]
        part = quantize(events, BPM, None, TS_4_4, "1/16", {"id": "P1", "name": "X", "clef": "treble"})
        self.assertGreaterEqual(len(part.measures), 2)
        m1_notes = [n for n in part.measures[0].notes if n.pitches]
        m2_notes = [n for n in part.measures[1].notes if n.pitches]
        self.assertEqual(len(m1_notes), 1)
        self.assertEqual(len(m2_notes), 1)
        self.assertTrue(m1_notes[0].tie_start)
        self.assertFalse(m1_notes[0].tie_stop)
        self.assertTrue(m2_notes[0].tie_stop)

    def test_grand_clef_splits_by_pitch(self):
        events = [
            NoteEvent(onset=0.0, offset=1.0, pitch=72),  # 高音譜
            NoteEvent(onset=0.0, offset=1.0, pitch=48),  # 低音譜（<60）
        ]
        part = quantize(events, BPM, None, TS_4_4, "1/16", {"id": "P1", "name": "Piano", "clef": "grand"})
        voice1 = [n for m in part.measures for n in m.notes if n.voice == 1 and n.pitches]
        voice2 = [n for m in part.measures for n in m.notes if n.voice == 2 and n.pitches]
        self.assertEqual(len(voice1), 1)
        self.assertEqual(voice1[0].pitches, [72])
        self.assertEqual(len(voice2), 1)
        self.assertEqual(voice2[0].pitches, [48])
        # 兩個 voice 都要各自補滿小節。
        for measure in part.measures:
            totals = _measure_total_by_voice(measure)
            self.assertEqual(totals.get(1), 1920)
            self.assertEqual(totals.get(2), 1920)

    def test_empty_input_grand_has_two_rest_voices(self):
        part = quantize([], BPM, None, TS_4_4, "1/16", {"id": "P1", "name": "Piano", "clef": "grand"})
        totals = _measure_total_by_voice(part.measures[0])
        self.assertEqual(totals.get(1), 1920)
        self.assertEqual(totals.get(2), 1920)

    def test_minimum_note_length_one_grid(self):
        # 極短音符（幾毫秒）量化後至少要有一格（1/16 = 120 tick）。
        events = [NoteEvent(onset=0.0, offset=0.01, pitch=60)]
        part = quantize(events, BPM, None, TS_4_4, "1/16", {"id": "P1", "name": "X", "clef": "treble"})
        note = [n for n in part.measures[0].notes if n.pitches][0]
        self.assertGreaterEqual(note.duration_tick, 120)

    def test_beat_times_mapping(self):
        # 提供明確的 beat_times（非等速），確認事件仍能正確落在對應拍位置。
        beat_times = [0.0, 0.6, 1.3, 1.9, 2.5]  # 不等速的 4 拍（quarter beat）
        events = [NoteEvent(onset=0.6, offset=1.3, pitch=60)]
        part = quantize(events, 100.0, beat_times, TS_4_4, "1/16", {"id": "P1", "name": "X", "clef": "treble"})
        note = [n for n in part.measures[0].notes if n.pitches][0]
        self.assertEqual(note.start_tick, 480)  # 第 2 拍起點 = 1 * 480 tick
        self.assertEqual(note.duration_tick, 480)  # 到第 3 拍，剛好 1 拍

    def test_notes_before_first_beat_are_shifted_not_collapsed(self):
        # 迴歸測試：第一個偵測到的拍點常常不是音檔裡第一個音符發生的時間
        # （例如節拍器抓到第一拍落在 0.8 秒，但第一個音符在 0 秒）。以前的做法
        # 是把換算出來的負 tick 逐一 clip 成 0，導致開頭好幾個不同時間的音符
        # 全部疊在同一個 tick 0 上；正確行為是整體平移，保留彼此的相對間隔。
        beat_times = [0.8, 1.2, 1.6, 2.0]  # 每拍 0.4 秒 = 150 BPM，第一拍在 0.8 秒
        events = [
            NoteEvent(onset=0.0, offset=0.35, pitch=60),
            NoteEvent(onset=0.4, offset=0.75, pitch=62),
            NoteEvent(onset=0.8, offset=1.15, pitch=64),
            NoteEvent(onset=1.2, offset=1.55, pitch=65),
        ]
        part = quantize(events, 150.0, beat_times, TS_4_4, "1/16",
                         {"id": "P1", "name": "X", "clef": "treble"})
        notes = [n for m in part.measures for n in m.notes if n.pitches]
        pitches_in_order = [n.pitches[0] for n in
                             sorted(notes, key=lambda n: n.start_tick)]
        # 四個音高都要留著、順序不變，不能被壓縮成同一個音。
        self.assertEqual(pitches_in_order, [60, 62, 64, 65])
        # 彼此間隔要維持等距（都相差一拍 = 480 tick），且第一個音符落在 tick 0。
        starts = [n.start_tick for n in sorted(notes, key=lambda n: n.start_tick)]
        self.assertEqual(starts[0], 0)
        diffs = [b - a for a, b in zip(starts, starts[1:])]
        self.assertEqual(diffs, [480, 480, 480])


class TestPreprocessing(unittest.TestCase):
    def test_remove_short_notes(self):
        events = [
            NoteEvent(onset=0.0, offset=0.02, pitch=60),  # 20ms，太短
            NoteEvent(onset=1.0, offset=1.5, pitch=62),   # 500ms
        ]
        kept = remove_short_notes(events, min_ms=60)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].pitch, 62)

    def test_merge_short_gaps(self):
        events = [
            NoteEvent(onset=0.0, offset=0.5, pitch=60),
            NoteEvent(onset=0.51, offset=1.0, pitch=60),  # 間隔 10ms，應合併
            NoteEvent(onset=2.0, offset=2.5, pitch=60),   # 間隔遠，保持獨立
        ]
        merged = merge_short_gaps(events, max_gap_ms=40.0)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0].onset, 0.0)
        self.assertAlmostEqual(merged[0].offset, 1.0)
        self.assertAlmostEqual(merged[1].onset, 2.0)


if __name__ == "__main__":
    unittest.main()
