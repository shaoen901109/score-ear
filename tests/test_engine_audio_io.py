# -*- coding: utf-8 -*-
import numpy as np
import pytest
import soundfile as sf

from engine.audio_io import AudioLoadError, duration_seconds, load_audio
from tests._audio_synth import sine_tone


def test_load_audio_wav_and_resample(tmp_path):
    sr_native = 44100
    y = sine_tone(440.0, 1.0, sr_native)
    wav_path = tmp_path / "tone.wav"
    sf.write(str(wav_path), y, sr_native)

    loaded, sr = load_audio(str(wav_path), target_sr=22050)

    assert sr == 22050
    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    # 重採樣後長度應約等於 1 秒 * 22050（容許誤差）
    assert abs(len(loaded) - 22050) < 200

    dur = duration_seconds(loaded, sr)
    assert abs(dur - 1.0) < 0.02


def test_load_audio_no_resample_when_same_sr(tmp_path):
    sr = 22050
    y = sine_tone(220.0, 0.5, sr)
    wav_path = tmp_path / "tone2.wav"
    sf.write(str(wav_path), y, sr)

    loaded, out_sr = load_audio(str(wav_path), target_sr=sr)
    assert out_sr == sr
    assert abs(len(loaded) - int(sr * 0.5)) < 10


def test_load_audio_missing_file_raises():
    with pytest.raises(AudioLoadError):
        load_audio("這個檔案不存在.wav")
