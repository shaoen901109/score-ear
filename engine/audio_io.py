# -*- coding: utf-8 -*-
"""音訊讀取與格式轉換：統一輸出單聲道 float32 numpy 陣列。

wav/flac/ogg 用 soundfile 直接讀取；其餘副檔名（mp3/m4a/mp4/aac/wma/webm 等）
先用 ffmpeg 轉成暫存 wav 再讀取。重採樣優先用 scipy.signal.resample_poly，
若失敗則退回 librosa.resample。
"""

from __future__ import annotations

import subprocess
import tempfile
from math import gcd
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from .paths import ffmpeg_exe

_NATIVE_EXTS = {".wav", ".flac", ".ogg"}


class AudioLoadError(RuntimeError):
    """音訊讀取或轉檔失敗（附中文說明）。"""


def _to_mono(data: np.ndarray) -> np.ndarray:
    """把可能是立體聲的陣列 (samples, channels) 轉成單聲道 (samples,)。"""
    data = np.asarray(data)
    if data.ndim == 1:
        return data.astype(np.float32)
    return np.mean(data, axis=1).astype(np.float32)


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """把單聲道音訊從 orig_sr 重採樣到 target_sr。"""
    y = np.asarray(y, dtype=np.float32)
    if orig_sr == target_sr or len(y) == 0:
        return y
    try:
        from scipy.signal import resample_poly

        g = gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        return resample_poly(y, up, down).astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def _decode_with_ffmpeg(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """呼叫 ffmpeg 把任意音訊/影片檔轉成暫存 wav（單聲道、target_sr）再讀回來。"""
    exe = ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "decoded.wav"
        cmd = [
            exe, "-y", "-i", path,
            "-ac", "1", "-ar", str(target_sr),
            "-vn", str(out_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not out_path.exists():
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise AudioLoadError(f"ffmpeg 轉檔失敗（{path}）：{stderr[-800:]}")
        y, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
        return _to_mono(y), int(sr)


def load_audio(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """讀取音訊檔，回傳 (單聲道 float32 陣列, 採樣率)，採樣率會重採樣到 target_sr。

    wav/flac/ogg 用 soundfile 直接讀；其他副檔名先用 ffmpeg 轉暫存 wav。
    """
    p = Path(path)
    if not p.exists():
        raise AudioLoadError(f"找不到音訊檔：{path}")

    ext = p.suffix.lower()
    try:
        if ext in _NATIVE_EXTS:
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
            y = _to_mono(y)
            sr = int(sr)
        else:
            y, sr = _decode_with_ffmpeg(str(p), target_sr)
    except AudioLoadError:
        raise
    except Exception as exc:
        raise AudioLoadError(f"讀取音訊檔失敗（{path}）：{exc}") from exc

    if sr != target_sr:
        y = resample(y, sr, target_sr)
        sr = target_sr

    return y.astype(np.float32), sr


def duration_seconds(y: np.ndarray, sr: int) -> float:
    """回傳音訊長度（秒）。"""
    if sr <= 0 or y is None:
        return 0.0
    return float(len(y)) / float(sr)
