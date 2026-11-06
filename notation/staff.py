# -*- coding: utf-8 -*-
"""Score -> 五線譜 SVG（每頁一個字串），內部透過 verovio 把 MusicXML 轉成 SVG。

verovio 可能還沒裝好（背景工作正在裝套件到 .venv），所以：
- import verovio 放在函式內，不在模組載入時執行。
- 沒裝好時丟出訊息清楚的 RuntimeError，呼叫端（app.server）可以接住並回報。

verovio 資源路徑（重要）：
- verovio 的 C++ 核心在讀取 Bravura/Leipzig 等字型資源時，遇到路徑含非 ASCII
  （中文/CJK）字元會讀取失敗（`tk.loadData()` 回傳 False，實測於本專案路徑
  「聽歌寫譜」）。開發路徑與未來 PyInstaller 打包路徑都可能含中文，因此不能
  直接用 `verovio.__file__` 旁邊的 `data/` 目錄。
- 修法：第一次呼叫時把 pip 套件內附的 `data/` 複製一份到
  `%LOCALAPPDATA%\\ScoreEar\\verovio-<version>\\`（純 ASCII，不用專案中文名
  「聽歌寫譜」本身的音譯或原文當目錄名，避免使用者帳號以外的中文成分），
  之後用 `tk.setResourcePath()` 指向這個快取複本。複製前用 `if not exists`
  檢查，避免每次啟動都重複複製。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .musicxml import score_to_musicxml
from .score_model import Score

PAGE_HEIGHT = 2970
PAGE_WIDTH = 2100
SCALE = 40


def _get_verovio_resource_dir(verovio_module) -> str:
    """回傳一份 ASCII 路徑下的 verovio 資源目錄，複製一次後快取。

    見本檔案頂端「verovio 資源路徑」說明：verovio 的 C++ 核心無法讀取含中文
    路徑下的字型資源，本專案的專案路徑與打包路徑都可能含中文，因此固定改用
    `%LOCALAPPDATA%\\ScoreEar\\verovio-<version>\\`。
    """
    src = Path(verovio_module.__file__).resolve().parent / "data"
    version = getattr(verovio_module, "__version__", None)
    if not version:
        try:
            from importlib.metadata import version as _pkg_version

            version = _pkg_version("verovio")
        except Exception:  # pragma: no cover - 極端環境才會走到
            version = "unknown"

    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata or not str(local_appdata).isascii():
        import tempfile

        local_appdata = tempfile.gettempdir()

    cache_dir = Path(local_appdata) / "ScoreEar" / f"verovio-{version}"
    if not cache_dir.exists():
        shutil.copytree(src, cache_dir)
    return str(cache_dir)


def score_to_staff_svg(score: Score) -> list:
    try:
        import verovio  # noqa: WPS433  (刻意延遲載入，見檔案頂端說明)
    except ImportError as exc:
        raise RuntimeError(
            "尚未安裝 verovio（五線譜渲染需要它）。"
            "請等背景套件安裝完成後再試一次，"
            "或手動執行：.venv\\Scripts\\python -m pip install verovio"
        ) from exc

    musicxml_str = score_to_musicxml(score)

    # initFont=False：避免建構子先用預設（可能含中文的）資源路徑嘗試載入字型
    # 印出一堆失敗訊息，改成明確呼叫 setResourcePath() 指向 ASCII 快取目錄。
    tk = verovio.toolkit(False)
    tk.setResourcePath(_get_verovio_resource_dir(verovio))
    tk.setOptions(
        json.dumps(
            {
                "pageHeight": PAGE_HEIGHT,
                "pageWidth": PAGE_WIDTH,
                "scale": SCALE,
                "adjustPageHeight": False,
            }
        )
    )
    ok = tk.loadData(musicxml_str)
    if not ok:
        raise RuntimeError("verovio 無法解析產生出來的 MusicXML，請檢查 score_to_musicxml() 的輸出。")

    page_count = tk.getPageCount()
    pages = []
    for i in range(1, page_count + 1):
        pages.append(tk.renderToSVG(i))
    return pages
