# -*- coding: utf-8 -*-
import numpy as np
import pytest

from tests._audio_synth import note_sequence

librosa = pytest.importorskip("librosa")

from engine.rhythm import detect_tempo_and_beats, detect_time_signature  # noqa: E402


def test_detect_tempo_c_e_g_120bpm():
    sr = 22050
    # C4 E4 G4 各 0.5 秒 @120bpm（四分音符 = 0.5 秒），重複多次讓節奏規律
    midis = [60, 64, 67] * 12
    y = note_sequence(midis, note_duration_s=0.5, sr=sr, attack_decay=True)

    bpm, beat_times, confidence = detect_tempo_and_beats(y, sr)

    # 允許偵測到 120 本身，或其倍拍/半拍
    candidates = [bpm, bpm * 2, bpm / 2]
    assert any(110.0 <= c <= 130.0 for c in candidates)
    assert 40.0 <= bpm <= 220.0
    assert isinstance(beat_times, np.ndarray)
    assert 0.0 <= confidence <= 1.0


def test_detect_tempo_empty_audio():
    bpm, beat_times, confidence = detect_tempo_and_beats(np.array([], dtype=np.float32), 22050)
    assert bpm == 120.0
    assert len(beat_times) == 0
    assert confidence == 0.0


# ---------------------------------------------------------------------------
# detect_time_signature：3/4 華爾滋 fixture 要判成 3/4，既有 4/4 fixture 不誤判
# ---------------------------------------------------------------------------

def test_detect_time_signature_waltz_is_3_4():
    import soundfile as sf

    from tests import make_fixtures

    fixtures = make_fixtures.generate_all()
    y, sr = sf.read(str(fixtures["waltz_mono"]["wav"]))
    y = y.astype(np.float32)

    bpm, beat_times, _confidence = detect_tempo_and_beats(y, sr)
    ts = detect_time_signature(y, sr, beat_times)

    assert ts == "3/4"


@pytest.mark.parametrize("fixture_name", ["scale_mono", "piano_chords", "multi_mix"])
def test_detect_time_signature_existing_4_4_fixtures_not_misclassified(fixture_name):
    import soundfile as sf

    from tests import make_fixtures

    fixtures = make_fixtures.generate_all()
    y, sr = sf.read(str(fixtures[fixture_name]["wav"]))
    y = y.astype(np.float32)

    bpm, beat_times, _confidence = detect_tempo_and_beats(y, sr)
    ts = detect_time_signature(y, sr, beat_times)

    assert ts == "4/4"


def test_detect_time_signature_too_few_beats_defaults_4_4():
    beat_times = np.array([0.0, 0.5, 1.0])
    ts = detect_time_signature(np.zeros(22050, dtype=np.float32), 22050, beat_times)
    assert ts == "4/4"


def test_detect_time_signature_empty_audio_defaults_4_4():
    ts = detect_time_signature(np.array([], dtype=np.float32), 22050, np.array([]))
    assert ts == "4/4"
