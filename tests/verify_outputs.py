# -*- coding: utf-8 -*-
"""輸出格式驗證：把 (a)(b)(c) 三個 fixture 的 Score 轉成 MusicXML / MIDI /
簡譜 SVG / 五線譜 SVG，存到 `runtime/out/`，並驗證：

- MusicXML：用 verovio 重新 loadData()，必須回傳 True。
- 簡譜/五線譜 SVG：用 `xml.etree.ElementTree` 解析必須成功。
- 印出簡譜 SVG 內出現的簡譜數字字元（1-7、升降記號、八度點），供人工確認
  C 大調音階是否正確顯示成 1 2 3 4 5 6 7 高音 1。

跟 tests/eval_accuracy.py 一樣不是 pytest 測試（會呼叫 basic-pitch/demucs/
piano 模型並寫檔到 runtime/），用
`.venv\\Scripts\\python -m tests.verify_outputs` 執行。
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Windows 終端機預設用 cp950（繁體中文 Big5）編碼，簡譜文字裡的音樂符號
# （例如「♩」）不在 cp950 字集內，直接 print 會丟 UnicodeEncodeError；
# 這裡把 stdout/stderr 重新設成 UTF-8（編碼失敗的字元用替代符號顯示，不中斷程式）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from tests import make_fixtures  # noqa: E402

OUT_DIR = REPO_ROOT / "runtime" / "out"


def _transcribe(wav_path: Path, mode: str, instrument_hint: str = "auto"):
    from engine.pipeline import transcribe
    from notation.score_model import TranscribeOptions

    options = TranscribeOptions(mode=mode, instrument_hint=instrument_hint, use_gpu=True)
    return transcribe(str(wav_path), options)


def _verify_musicxml_reloads_in_verovio(xml_str: str) -> bool:
    import verovio

    from notation.staff import _get_verovio_resource_dir

    tk = verovio.toolkit(False)
    tk.setResourcePath(_get_verovio_resource_dir(verovio))
    return bool(tk.loadData(xml_str))


def _verify_svg_parses(svg_str: str) -> bool:
    try:
        ET.fromstring(svg_str)
        return True
    except ET.ParseError:
        return False


def _extract_jianpu_digits(svg_str: str) -> str:
    """粗略抓出簡譜 SVG 裡 <text> 節點的內容，方便人工檢查數字/升降記號是否合理。"""
    texts = re.findall(r"<text[^>]*>(.*?)</text>", svg_str, flags=re.DOTALL)
    cleaned = [re.sub(r"<[^>]+>", "", t).strip() for t in texts]
    return " ".join(t for t in cleaned if t)


def process_fixture(name: str, wav_path: Path, mode: str, instrument_hint: str = "auto") -> dict:
    from notation.jianpu import score_to_jianpu_svg
    from notation.midi import score_to_midi_bytes
    from notation.musicxml import score_to_musicxml
    from notation.staff import score_to_staff_svg

    print(f"-- {name} ({mode}/{instrument_hint}) --")
    score = _transcribe(wav_path, mode, instrument_hint)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_str = score_to_musicxml(score)
    xml_path = OUT_DIR / f"{name}.musicxml"
    xml_path.write_text(xml_str, encoding="utf-8")
    xml_ok = _verify_musicxml_reloads_in_verovio(xml_str)
    print(f"  musicxml -> {xml_path}  verovio_reload_ok={xml_ok}")

    midi_bytes = score_to_midi_bytes(score)
    midi_path = OUT_DIR / f"{name}.mid"
    midi_path.write_bytes(midi_bytes)
    print(f"  midi -> {midi_path} ({len(midi_bytes)} bytes)")

    jianpu_svg = score_to_jianpu_svg(score)
    jianpu_path = OUT_DIR / f"{name}.jianpu.svg"
    jianpu_path.write_text(jianpu_svg, encoding="utf-8")
    jianpu_ok = _verify_svg_parses(jianpu_svg)
    jianpu_text = _extract_jianpu_digits(jianpu_svg)
    print(f"  jianpu.svg -> {jianpu_path}  xml_parse_ok={jianpu_ok}")
    print(f"  jianpu 文字內容：{jianpu_text}")

    staff_pages = score_to_staff_svg(score)
    staff_ok_all = True
    for i, page_svg in enumerate(staff_pages, start=1):
        ok = _verify_svg_parses(page_svg)
        staff_ok_all = staff_ok_all and ok
        print(f"  staff.svg page {i}: xml_parse_ok={ok} len={len(page_svg)}")
    staff_zip_path = OUT_DIR / f"{name}.staff.zip"
    with zipfile.ZipFile(staff_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, page_svg in enumerate(staff_pages, start=1):
            zf.writestr(f"staff_{i:02d}.svg", page_svg.encode("utf-8"))
    print(f"  staff pages zipped -> {staff_zip_path}")

    return {
        "name": name,
        "xml_ok": xml_ok,
        "jianpu_ok": jianpu_ok,
        "jianpu_text": jianpu_text,
        "staff_ok": staff_ok_all,
    }


def main() -> None:
    fixtures = make_fixtures.generate_all()

    results = []
    results.append(process_fixture("scale_mono", fixtures["scale_mono"]["wav"], "solo", "vocal"))
    results.append(process_fixture("piano_chords", fixtures["piano_chords"]["wav"], "solo", "piano"))
    results.append(process_fixture("multi_mix", fixtures["multi_mix"]["wav"], "multi", "auto"))

    print("\n== 總結 ==")
    all_ok = True
    for r in results:
        ok = r["xml_ok"] and r["jianpu_ok"] and r["staff_ok"]
        all_ok = all_ok and ok
        print(f"{r['name']}: musicxml_reload={r['xml_ok']} jianpu_xml={r['jianpu_ok']} "
              f"staff_xml={r['staff_ok']}  -> {'OK' if ok else 'FAIL'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
