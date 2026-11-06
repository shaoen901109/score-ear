# -*- coding: utf-8 -*-
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("demucs")

pytestmark = pytest.mark.slow


def _has_local_htdemucs_checkpoint() -> bool:
    # demucs 4.x 透過 huggingface_hub 下載模型（見 engine/separate.py 的說明），
    # 不再放進 torch hub 的 checkpoints 資料夾，所以除了舊路徑也要檢查
    # huggingface hub 的快取目錄，否則模型明明已經下載好，這個判斷式還是會
    # 一直回傳 False，測試永遠被跳過。
    cache_dirs = [
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Roaming\torch\hub\checkpoints"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expandvars(r"%USERPROFILE%\.cache\huggingface\hub"),
    ]
    for d in cache_dirs:
        if os.path.isdir(d):
            for root, _dirs, files in os.walk(d):
                for f in files:
                    if "demucs" in f.lower() or "htdemucs" in root.lower():
                        return True
    return False


def test_get_model_loads_htdemucs_when_cached():
    if not _has_local_htdemucs_checkpoint():
        pytest.skip("htdemucs 模型尚未下載到本機快取，測試不得依賴網路，略過")
    from engine.separate import _get_model

    model, device = _get_model(use_gpu=False)
    assert model is not None
    assert device == "cpu"
