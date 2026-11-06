# -*- coding: utf-8 -*-
"""路徑輔助：應用程式根目錄、模型資料夾、ffmpeg 執行檔位置。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def app_root() -> Path:
    """回傳應用程式根目錄。

    - PyInstaller 打包（frozen）時：回傳 exe 所在目錄（onedir 模式下模型隨資料夾附帶）。
    - 開發模式：回傳專案根目錄（本檔案往上兩層：engine/ -> 專案根）。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    """回傳模型檔存放目錄（不存在則自動建立）。"""
    d = app_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ffmpeg_exe() -> str:
    """尋找可用的 ffmpeg 執行檔路徑。

    優先順序：
    1. 系統 PATH 內的 ffmpeg
    2. imageio-ffmpeg 套件內建的 ffmpeg

    兩者都找不到時，丟出附中文說明的 RuntimeError。
    """
    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass

    raise RuntimeError(
        "找不到 ffmpeg 執行檔：系統 PATH 中沒有安裝 ffmpeg，"
        "且 imageio-ffmpeg 套件也未提供可用的執行檔。"
        "請安裝 ffmpeg 並加入 PATH，或確認 imageio-ffmpeg 套件安裝正確。"
    )
