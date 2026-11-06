# -*- coding: utf-8 -*-
"""鋼琴轉譜：使用 piano_transcription_inference（ByteDance）模型。"""

from __future__ import annotations

import os
import urllib.request
from typing import Callable, Dict, List, Optional

import numpy as np

from notation.score_model import NoteEvent

from .paths import models_dir

_model_cache: Dict[str, object] = {}

_CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
_CHECKPOINT_MIN_SIZE = int(1.6e8)  # 與 piano_transcription_inference 套件內建的檢查一致
_TARGET_SR = 16000
_LEAD_SILENCE_S = 0.3  # 見 transcribe_piano() 內的說明


class PianoTranscribeError(RuntimeError):
    """鋼琴轉譜失敗（附中文說明）。"""


def _checkpoint_path() -> str:
    """回傳鋼琴轉譜模型 checkpoint 路徑；資料夾不存在則建立。"""
    d = models_dir() / "piano_transcription"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / _CHECKPOINT_NAME)


def _ensure_checkpoint_downloaded(path: str) -> None:
    """確保 checkpoint 檔案存在，不存在（或不完整）就自己下載。

    piano_transcription_inference 套件本身遇到檔案不存在時，是用
    `os.system('wget -O ... ...')` 下載（見它的 `inference.py`）：Windows 環境
    預設沒有 `wget` 指令，`os.system` 呼叫會直接失敗（且不會拋例外、只是
    靜默地什麼都沒做），套件接著呼叫 `torch.load()` 就會丟出令人困惑的
    `FileNotFoundError`，看起來像「下載中」但其實根本沒開始下載。這裡改成
    自己用標準庫 `urllib` 下載（不依賴系統是否裝了 wget），檔案存在且大小
    合理（沿用套件同一個 1.6e8 bytes 門檻）就跳過。
    """
    if os.path.exists(path) and os.path.getsize(path) >= _CHECKPOINT_MIN_SIZE:
        return

    tmp_path = path + ".part"
    try:
        with urllib.request.urlopen(_CHECKPOINT_URL, timeout=60) as resp, open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise PianoTranscribeError(
            f"下載鋼琴轉譜模型 checkpoint 失敗（{_CHECKPOINT_URL}）：{exc}"
        ) from exc

    if os.path.getsize(tmp_path) < _CHECKPOINT_MIN_SIZE:
        os.remove(tmp_path)
        raise PianoTranscribeError(
            f"下載鋼琴轉譜模型 checkpoint 不完整（檔案過小），請檢查網路後重試：{_CHECKPOINT_URL}"
        )
    os.replace(tmp_path, path)


def _get_transcriptor(use_gpu: bool):
    import torch
    from piano_transcription_inference import PianoTranscription

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    key = f"piano:{device}"
    if key not in _model_cache:
        checkpoint_path = _checkpoint_path()
        _ensure_checkpoint_downloaded(checkpoint_path)
        _model_cache[key] = PianoTranscription(device=device, checkpoint_path=checkpoint_path)
    return _model_cache[key]


def transcribe_piano(
    y: np.ndarray,
    sr: int,
    use_gpu: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[NoteEvent]:
    """用鋼琴轉譜模型辨識音符，回傳 NoteEvent 清單（秒為單位）。

    piano_transcription_inference 內部固定用 16000Hz，取樣率不符時會先重採樣。
    GPU 執行遇到 CUDA 相關錯誤時，自動退回 CPU 重試一次。
    """
    if progress:
        progress(0.0, "載入鋼琴轉譜模型")

    try:
        transcriptor = _get_transcriptor(use_gpu)
        device = "cuda" if (use_gpu and _cuda_available()) else "cpu"
    except Exception as exc:
        raise PianoTranscribeError(f"載入鋼琴轉譜模型失敗：{exc}") from exc

    y = np.asarray(y, dtype=np.float32)
    if sr != _TARGET_SR:
        from .audio_io import resample

        y = resample(y, sr, _TARGET_SR)

    # 在音檔開頭補一小段靜音再送進模型，辨識完再把補進去的時長減回去。
    #
    # 實測發現：如果音符從第 0 秒就直接開始（沒有任何前導靜音／情境），
    # piano_transcription_inference 常常完全偵測不到第一個音符的 onset
    # （或是把它偵測成晚了將近一整秒的錯誤位置），對只有 3-4 秒的短音檔
    # 影響特別明顯；補一段前導靜音讓模型有機會先建立好內部狀態後，
    # 第一個音符就能準確定位在補進去的靜音結束處。
    lead_pad_samples = int(round(_LEAD_SILENCE_S * _TARGET_SR))
    y_padded = np.concatenate([np.zeros(lead_pad_samples, dtype=np.float32), y])

    try:
        if progress:
            progress(0.3, "辨識鋼琴音符中")
        result = transcriptor.transcribe(y_padded, None)
    except RuntimeError as exc:
        if device == "cuda":
            if progress:
                progress(0.3, "GPU 辨識失敗，改用 CPU 重試")
            try:
                transcriptor = _get_transcriptor(False)
                result = transcriptor.transcribe(y_padded, None)
            except Exception as exc2:
                raise PianoTranscribeError(f"鋼琴轉譜失敗（CPU 重試後仍失敗）：{exc2}") from exc2
        else:
            raise PianoTranscribeError(f"鋼琴轉譜失敗：{exc}") from exc
    except Exception as exc:
        raise PianoTranscribeError(f"鋼琴轉譜失敗：{exc}") from exc

    events: List[NoteEvent] = []
    for note in result.get("est_note_events", []):
        onset = float(note["onset_time"]) - _LEAD_SILENCE_S
        offset = float(note["offset_time"]) - _LEAD_SILENCE_S
        if onset < 0.0:
            onset = 0.0
        if offset <= onset:
            offset = onset + 0.05
        pitch = int(note["midi_note"])
        velocity = int(np.clip(float(note.get("velocity", 80)), 1, 127))
        events.append(NoteEvent(onset=onset, offset=offset, pitch=pitch, velocity=velocity, confidence=1.0))

    events.sort(key=lambda e: e.onset)
    if progress:
        progress(1.0, "鋼琴轉譜完成")
    return events


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
