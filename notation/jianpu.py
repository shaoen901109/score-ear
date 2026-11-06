# -*- coding: utf-8 -*-
"""Score -> 簡譜 SVG（純字串組裝，不依賴任何繪圖/排版外部庫）。

版面設計（在符合 docs/ARCHITECTURE.md 規則的前提下，做了以下實作簡化，均不影響
可讀性與正確性，列在報告中供其他模組作者參考）：
- 每個音符格採「固定寬度」排版（不像真正的活版印刷依時值/字寬動態調整），
  版面因此比較規則、程式也單純很多。
- 每行固定 4 小節；一行內小節若音符很多，畫面寬度會自動比 794px 的
  預設寬度再加寬，以避免音符互相重疊或被裁切（不做縮放，維持字體可讀）。
- 和弦的八度點改畫在該音符數字的右上角（一個小點堆疊），而不是整個和弦
  正上方/正下方，避免跟旁邊的音符打架。
- 跨行（一小節在某一行結尾、下一個音在下一行開頭）的圓滑線暫不繪製弧線
  （沒有畫面空間可畫），但音符本身、tie 語意仍正確產生，只是少了視覺弧線。
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from .musicxml import _duration_to_components
from .score_model import Score

PAGE_WIDTH = 794
MARGIN_X = 40
MARGIN_TOP = 50
CELL_WIDTH = 34
ROW_HEIGHT = 100
MEASURES_PER_ROW = 4
BAR_GAP = 14

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# 半音 -> (簡譜級數, 升降記號)，以大調音階為準（1=Do）。
_SEMITONE_TABLE = {
    0: (1, ""), 1: (1, "#"), 2: (2, ""), 3: (2, "#"), 4: (3, ""),
    5: (4, ""), 6: (4, "#"), 7: (5, ""), 8: (5, "#"), 9: (6, ""),
    10: (6, "#"), 11: (7, ""),
}

_DRUM_SYMBOL = {
    36: ("X", 8),   # kick，放低
    38: ("O", 0),   # snare，基準線
    42: ("x", -8),  # hihat，放高
}

_TYPE_UNDERLINES = {"quarter": 0, "eighth": 1, "16th": 2, "32nd": 3, "half": 0, "whole": 0}
_TYPE_DASHES = {"whole": 3, "half": 1}  # 延長線數量（附加在數字之後）


def _tonic_to_pc(tonic: str) -> int:
    tonic = (tonic or "C").strip()
    if not tonic:
        return 0
    letter = tonic[0].upper()
    pc = _LETTER_PC.get(letter, 0)
    for ch in tonic[1:]:
        if ch in ("#", "♯"):
            pc += 1
        elif ch in ("b", "♭"):
            pc -= 1
    return pc % 12


def _pitch_to_degree(pitch: int, tonic_pc: int):
    """回傳 (degree 1-7, accidental '#'/''，octave_index 正上負下)。"""
    ref = tonic_pc + 60
    diff = pitch - ref
    octave_index = diff // 12
    semitone = diff % 12
    degree, accidental = _SEMITONE_TABLE[semitone]
    return degree, accidental, octave_index


class _Glyph:
    """一個「格子」：可能是單音、和弦、休止符或鼓的打擊記號。"""

    __slots__ = ("labels", "underline", "dashes", "dots", "tie_prev", "tie_next", "is_rest")

    def __init__(self, labels, underline, dashes, dots, tie_prev, tie_next, is_rest):
        self.labels = labels          # list[str]，由下而上堆疊（和弦時多個）
        self.underline = underline
        self.dashes = dashes
        self.dots = dots
        self.tie_prev = tie_prev
        self.tie_next = tie_next
        self.is_rest = is_rest


def _build_glyphs_for_note(note, tonic_pc: int, is_percussion: bool):
    """把一個量化後的 Note 拆成一串 _Glyph（時值拆分時彼此用 tie 相接）。"""
    components = _duration_to_components(note.duration_tick)
    k = len(components)
    glyphs = []
    is_rest = len(note.pitches) == 0

    for i, (type_name, dots, _ticks) in enumerate(components):
        if k == 1:
            tie_prev, tie_next = note.tie_stop, note.tie_start
        else:
            tie_prev = note.tie_stop if i == 0 else True
            tie_next = note.tie_start if i == k - 1 else True

        underline = _TYPE_UNDERLINES.get(type_name, 0)
        dashes = _TYPE_DASHES.get(type_name, 0)

        if is_rest:
            labels = ["0"]
        elif is_percussion:
            labels = []
            for p in note.pitches:
                sym, _shift = _DRUM_SYMBOL.get(p, ("O", 0))
                labels.append(sym)
        else:
            labels = []
            for p in sorted(note.pitches):
                degree, accidental, octave_index = _pitch_to_degree(p, tonic_pc)
                dot_mark = "·" * min(abs(octave_index), 3)
                text = f"{accidental}{degree}"
                if octave_index > 0:
                    text += f"^{dot_mark}"
                elif octave_index < 0:
                    text += f"_{dot_mark}"
                labels.append(text)

        glyphs.append(_Glyph(labels, underline, dashes, dots, tie_prev, tie_next, is_rest))

    return glyphs


def _render_glyph(x: int, baseline: int, glyph: "_Glyph") -> str:
    parts = []
    n = len(glyph.labels)
    for idx, label in enumerate(glyph.labels):
        dy = baseline - idx * 16
        parts.append(
            f'<text x="{x}" y="{dy}" font-family="sans-serif" font-size="16" '
            f'text-anchor="middle">{escape(label)}</text>'
        )
    text_right = x + 8 + 6 * max((len(l) for l in glyph.labels), default=1)
    if glyph.dots:
        parts.append(
            f'<text x="{text_right}" y="{baseline}" font-family="sans-serif" '
            f'font-size="16">{"." * glyph.dots}</text>'
        )
    if glyph.dashes:
        dash_str = " - " * glyph.dashes
        parts.append(
            f'<text x="{x + 16}" y="{baseline}" font-family="sans-serif" '
            f'font-size="14">{escape(dash_str)}</text>'
        )
    for u in range(glyph.underline):
        y = baseline + 6 + u * 4
        parts.append(
            f'<line x1="{x - 10}" y1="{y}" x2="{x + 10}" y2="{y}" stroke="black" stroke-width="1"/>'
        )
    return "".join(parts)


def _tie_arc(x1: int, x2: int, y: int) -> str:
    mx = (x1 + x2) / 2
    top = y - 16
    return f'<path d="M {x1} {y} Q {mx} {top} {x2} {y}" stroke="black" fill="none" stroke-width="1"/>'


def _render_part(part, score: Score, y_start: int) -> tuple:
    """回傳 (svg_fragment, 佔用高度, 此段實際用到的最大 x 座標)。"""
    tonic_pc = _tonic_to_pc(score.key.tonic)
    is_percussion = part.clef == "percussion"

    fragments = [
        f'<text x="{MARGIN_X}" y="{y_start}" font-family="sans-serif" font-size="18" '
        f'font-weight="bold">{escape(part.name)}</text>'
    ]
    header_y = y_start + 24
    header_text = (
        f"1={escape(score.key.tonic or 'C')}  "
        f"{score.time_signature.beats}/{score.time_signature.beat_type}  "
        f"♩={int(round(score.bpm))}"
    )
    fragments.append(
        f'<text x="{MARGIN_X}" y="{header_y}" font-family="sans-serif" font-size="14">'
        f"{header_text}</text>"
    )

    cursor_y = header_y + 20

    # 攤平成 glyph 序列，並記錄每個 glyph 屬於第幾小節（用來畫小節線）。
    glyph_measures = []  # list[(measure_no, glyph)]
    for measure in sorted(part.measures, key=lambda m: m.number):
        notes = sorted(measure.notes, key=lambda n: n.start_tick)
        # 只取 voice 1（grand 的 voice 2 屬於低音譜，簡譜以高音譜/主旋律為準）。
        notes = [n for n in notes if n.voice == 1] or notes
        for note in notes:
            for g in _build_glyphs_for_note(note, tonic_pc, is_percussion):
                glyph_measures.append((measure.number, g))

    if not glyph_measures:
        glyph_measures = [(1, _Glyph(["0"], 0, 0, 0, False, False, True))]

    # 依小節分組
    measures_dict = {}
    for m_no, g in glyph_measures:
        measures_dict.setdefault(m_no, []).append(g)
    measure_numbers = sorted(measures_dict.keys())

    row_chunks = [
        measure_numbers[i : i + MEASURES_PER_ROW]
        for i in range(0, len(measure_numbers), MEASURES_PER_ROW)
    ] or [[]]

    max_x = MARGIN_X

    for row in row_chunks:
        x = MARGIN_X
        baseline = cursor_y + 40
        row_svg = [f'<line x1="{x}" y1="{baseline}" x2="{x}" y2="{baseline - 24}" stroke="black" stroke-width="1"/>']
        prev_anchor = None
        for m_no in row:
            for g in measures_dict[m_no]:
                row_svg.append(_render_glyph(x, baseline, g))
                if g.tie_prev and prev_anchor is not None:
                    row_svg.append(_tie_arc(prev_anchor[0], x, baseline - 20))
                prev_anchor = (x, baseline) if g.tie_next else None
                x += CELL_WIDTH
            x += BAR_GAP / 2
            row_svg.append(
                f'<line x1="{x}" y1="{baseline}" x2="{x}" y2="{baseline - 24}" stroke="black" stroke-width="1"/>'
            )
            x += BAR_GAP / 2
        max_x = max(max_x, x)
        fragments.append("".join(row_svg))
        cursor_y += ROW_HEIGHT

    used_height = cursor_y - y_start + 20
    return "".join(fragments), used_height, max_x


def score_to_jianpu_svg(score: Score, part_index=None) -> str:
    parts = score.parts or []
    if part_index is not None:
        if 0 <= part_index < len(parts):
            parts = [parts[part_index]]
        else:
            parts = []

    fragments = []
    y = MARGIN_TOP
    title = score.title or "聽歌寫譜"
    fragments.append(
        f'<text x="{PAGE_WIDTH / 2}" y="{y}" font-family="sans-serif" font-size="22" '
        f'text-anchor="middle" font-weight="bold">{escape(title)}</text>'
    )
    y += 30

    max_x = PAGE_WIDTH - MARGIN_X
    if not parts:
        fragments.append(
            f'<text x="{MARGIN_X}" y="{y + 20}" font-family="sans-serif" font-size="14">'
            "（無音符）</text>"
        )
        y += 60
    else:
        for part in parts:
            frag, height, part_max_x = _render_part(part, score, y)
            fragments.append(frag)
            y += height
            max_x = max(max_x, part_max_x)

    total_height = y + MARGIN_TOP
    width = max(PAGE_WIDTH, int(max_x) + MARGIN_X)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" '
        f'viewBox="0 0 {width} {total_height}" font-family="sans-serif">'
        f'<rect x="0" y="0" width="{width}" height="{total_height}" fill="white"/>'
        + "".join(fragments)
        + "</svg>"
    )
    return svg
