# -*- coding: utf-8 -*-
"""notation.score_model 的序列化 round-trip 測試。"""

import unittest

from notation.score_model import (
    KeySignature,
    Measure,
    Note,
    NoteEvent,
    Part,
    Score,
    TimeSignature,
    TranscribeOptions,
)


def _make_score() -> Score:
    return Score(
        title="測試曲",
        bpm=120.0,
        time_signature=TimeSignature(beats=4, beat_type=4),
        key=KeySignature(fifths=0, mode="major", tonic="C"),
        parts=[
            Part(
                id="P1",
                name="Piano",
                instrument="piano",
                clef="grand",
                midi_program=0,
                measures=[
                    Measure(
                        number=1,
                        notes=[
                            Note(start_tick=0, duration_tick=480, pitches=[60, 64, 67], voice=1),
                            Note(start_tick=480, duration_tick=480, pitches=[], voice=1),
                            Note(start_tick=960, duration_tick=960, pitches=[48], voice=2, tie_start=True),
                        ],
                    ),
                    Measure(
                        number=2,
                        notes=[
                            Note(start_tick=0, duration_tick=1920, pitches=[], voice=1),
                            Note(start_tick=0, duration_tick=1920, pitches=[48], voice=2, tie_stop=True),
                        ],
                    ),
                ],
            )
        ],
        source_file="foo.wav",
        warnings=["速度偵測信心低"],
    )


class TestNoteEvent(unittest.TestCase):
    def test_round_trip(self):
        ev = NoteEvent(onset=1.5, offset=2.25, pitch=60, velocity=100, confidence=0.9)
        d = ev.to_dict()
        ev2 = NoteEvent.from_dict(d)
        self.assertEqual(ev, ev2)

    def test_defaults(self):
        ev = NoteEvent.from_dict({"onset": 0.0, "offset": 1.0, "pitch": 64})
        self.assertEqual(ev.velocity, 80)
        self.assertEqual(ev.confidence, 1.0)


class TestKeyAndTimeSignature(unittest.TestCase):
    def test_key_round_trip(self):
        k = KeySignature(fifths=-3, mode="minor", tonic="Eb")
        self.assertEqual(KeySignature.from_dict(k.to_dict()), k)

    def test_time_round_trip(self):
        t = TimeSignature(beats=6, beat_type=8)
        self.assertEqual(TimeSignature.from_dict(t.to_dict()), t)


class TestNoteMeasurePart(unittest.TestCase):
    def test_note_round_trip(self):
        n = Note(start_tick=120, duration_tick=240, pitches=[60, 64], tie_start=True, voice=2)
        n2 = Note.from_dict(n.to_dict())
        self.assertEqual(n, n2)

    def test_measure_round_trip(self):
        m = Measure(number=3, notes=[Note(start_tick=0, duration_tick=1920, pitches=[])])
        m2 = Measure.from_dict(m.to_dict())
        self.assertEqual(m.number, m2.number)
        self.assertEqual(m.notes, m2.notes)

    def test_part_round_trip(self):
        p = Part(
            id="P2",
            name="Drums",
            instrument="drums",
            clef="percussion",
            midi_program=0,
            measures=[Measure(number=1, notes=[Note(start_tick=0, duration_tick=1920, pitches=[36])])],
        )
        p2 = Part.from_dict(p.to_dict())
        self.assertEqual(p.id, p2.id)
        self.assertEqual(p.measures, p2.measures)


class TestScore(unittest.TestCase):
    def test_to_dict_from_dict_round_trip(self):
        score = _make_score()
        d = score.to_dict()
        score2 = Score.from_dict(d)
        self.assertEqual(score.title, score2.title)
        self.assertEqual(score.bpm, score2.bpm)
        self.assertEqual(score.time_signature, score2.time_signature)
        self.assertEqual(score.key, score2.key)
        self.assertEqual(score.source_file, score2.source_file)
        self.assertEqual(score.warnings, score2.warnings)
        self.assertEqual(len(score.parts), len(score2.parts))
        self.assertEqual(score.parts[0].measures, score2.parts[0].measures)

    def test_to_json_from_json_round_trip(self):
        score = _make_score()
        s = score.to_json()
        self.assertIsInstance(s, str)
        score2 = Score.from_json(s)
        self.assertEqual(score.to_dict(), score2.to_dict())

    def test_dict_is_json_serializable(self):
        import json

        score = _make_score()
        # 不應該丟例外：確保 to_dict() 的結果是純 dict/list/str/number。
        json.dumps(score.to_dict(), ensure_ascii=False)


class TestTranscribeOptions(unittest.TestCase):
    def test_defaults_round_trip(self):
        opt = TranscribeOptions()
        opt2 = TranscribeOptions.from_dict(opt.to_dict())
        self.assertEqual(opt, opt2)

    def test_custom_round_trip(self):
        opt = TranscribeOptions(
            mode="multi",
            instrument_hint="piano",
            bpm_override=96.5,
            time_signature="3/4",
            min_note_ms=80,
            quantize="1/32",
            use_gpu=False,
        )
        opt2 = TranscribeOptions.from_dict(opt.to_dict())
        self.assertEqual(opt, opt2)

    def test_bpm_override_none(self):
        opt = TranscribeOptions.from_dict({})
        self.assertIsNone(opt.bpm_override)


if __name__ == "__main__":
    unittest.main()
