# -*- coding: utf-8 -*-
"""音源分離：demucs htdemucs，回傳 vocals/drums/bass/other 各聲部的單聲道音訊。

demucs 需要 44100Hz 立體聲輸入；本模組自行處理重採樣與單聲道轉立體聲（左右複製同一份）。
輸出的每個 stem 會重採樣回輸入時的原始取樣率，維持單聲道 float32。
模型延遲載入並在模組層依裝置（cuda/cpu）快取。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

_DEMUCS_SR = 44100

_model_cache: Dict[str, object] = {}


class SeparateError(RuntimeError):
    """音源分離失敗（附中文說明）。"""


def _get_model(use_gpu: bool):
    import torch
    from demucs.pretrained import get_model

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    key = f"htdemucs:{device}"
    if key not in _model_cache:
        model = get_model("htdemucs")
        model.to(device)
        model.eval()
        _model_cache[key] = model
    return _model_cache[key], device


def separate(
    y: np.ndarray,
    sr: int,
    use_gpu: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, np.ndarray]:
    """用 demucs htdemucs 分離人聲/鼓/貝斯/其他。

    輸入單聲道任意取樣率，內部轉成 demucs 需要的 44100 立體聲（左右聲道複製）。
    回傳 {stem 名稱: 單聲道 float32 陣列}，取樣率與輸入的 sr 相同。
    GPU 執行遇到錯誤時，自動退回 CPU 重試一次。
    """
    import torch

    from .audio_io import resample

    if progress:
        progress(0.0, "載入音源分離模型")

    try:
        model, device = _get_model(use_gpu)
    except Exception as exc:
        raise SeparateError(f"載入音源分離模型失敗：{exc}") from exc

    y44 = resample(np.asarray(y, dtype=np.float32), sr, _DEMUCS_SR)
    stereo = np.stack([y44, y44], axis=0)  # (2, samples)

    def _run(model_, device_) -> np.ndarray:
        from demucs.apply import apply_model

        wav = torch.from_numpy(stereo.copy()).float().unsqueeze(0).to(device_)
        with torch.no_grad():
            sources = apply_model(model_, wav, device=device_, progress=False)[0]
        return sources.cpu().numpy()

    try:
        if progress:
            progress(0.3, "分離音源中（可能需要數分鐘）")
        sources = _run(model, device)
    except RuntimeError as exc:
        if device == "cuda":
            if progress:
                progress(0.3, "GPU 分離失敗，改用 CPU 重試")
            try:
                model, device = _get_model(False)
                sources = _run(model, "cpu")
            except Exception as exc2:
                raise SeparateError(f"音源分離失敗（CPU 重試後仍失敗）：{exc2}") from exc2
        else:
            raise SeparateError(f"音源分離失敗：{exc}") from exc
    except Exception as exc:
        raise SeparateError(f"音源分離失敗：{exc}") from exc

    result: Dict[str, np.ndarray] = {}
    source_names = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))
    for idx, name in enumerate(source_names):
        stem_stereo = sources[idx]  # (2, samples) @ 44100
        mono44 = np.mean(stem_stereo, axis=0).astype(np.float32)
        result[name] = resample(mono44, _DEMUCS_SR, sr)

    if progress:
        progress(1.0, "音源分離完成")
    return result
