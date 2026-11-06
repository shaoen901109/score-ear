# -*- coding: utf-8 -*-
"""notation.staff 測試。

verovio 在開發環境可能還沒裝好（另一個背景工作正在裝套件），所以：
- 「缺套件時丟出明確 RuntimeError」這個測試用 sys.modules 強制模擬缺套件，
  不管當下環境實際有沒有裝 verovio 都能穩定跑。
- 真的呼叫 verovio 渲染的測試，用 pytest.importorskip 在沒裝好時跳過
  （若用 unittest 執行，改用 try/except 手動 skip）。
"""

import sys
import unittest
from unittest import mock

from notation.quantize import quantize
from notation.score_model import KeySignature, Score, TimeSignature

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

TS_4_4 = TimeSignature(beats=4, beat_type=4)


def _simple_score() -> Score:
    part = quantize([], 120.0, None, TS_4_4, "1/16", {
        "id": "P1", "name": "Piano", "clef": "treble", "midi_program": 0,
    })
    return Score(title="Test", bpm=120.0, time_signature=TS_4_4,
                 key=KeySignature(fifths=0, mode="major", tonic="C"), parts=[part])


class TestStaffMissingVerovio(unittest.TestCase):
    def test_raises_clear_runtime_error_when_verovio_missing(self):
        from notation import staff as staff_module

        with mock.patch.dict(sys.modules, {"verovio": None}):
            with self.assertRaises(RuntimeError) as ctx:
                staff_module.score_to_staff_svg(_simple_score())
            self.assertIn("verovio", str(ctx.exception))


if pytest is not None:

    def test_real_render_when_verovio_available():
        pytest.importorskip("verovio")
        from notation.staff import score_to_staff_svg

        pages = score_to_staff_svg(_simple_score())
        assert isinstance(pages, list)
        assert len(pages) >= 1
        assert "<svg" in pages[0]
        # 真正渲染出音符/圖形元素（不是只有空殼 SVG）：verovio 在中文路徑下
        # 資源載入失敗時仍會回傳 loadData=False 或近乎空的 SVG，見
        # notation/staff.py 的 _get_verovio_resource_dir()。
        assert "<path" in pages[0] or "<g " in pages[0] or "<g>" in pages[0]

    def test_verovio_resource_dir_is_ascii_and_cached():
        pytest.importorskip("verovio")
        import verovio as verovio_module

        from notation.staff import _get_verovio_resource_dir

        resource_dir = _get_verovio_resource_dir(verovio_module)
        assert resource_dir.isascii()
        assert "ScoreEar" in resource_dir
        from pathlib import Path

        assert Path(resource_dir).exists()
        assert (Path(resource_dir) / "Bravura.xml").exists()

else:  # pragma: no cover - 沒有 pytest 時的 unittest 備援

    class TestStaffRealRender(unittest.TestCase):
        def test_real_render_when_verovio_available(self):
            try:
                import verovio  # noqa: F401
            except ImportError:
                self.skipTest("verovio 尚未安裝")
            from notation.staff import score_to_staff_svg

            pages = score_to_staff_svg(_simple_score())
            self.assertIsInstance(pages, list)
            self.assertGreaterEqual(len(pages), 1)


if __name__ == "__main__":
    unittest.main()
