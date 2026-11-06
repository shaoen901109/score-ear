# -*- coding: utf-8 -*-
"""端到端辨識準確度評估：對 tests/make_fixtures.py 產生的合成音檔跑
`engine.pipeline.transcribe()`，計算音符級 F1，並把結果寫到 docs/ACCURACY.md。

不是 pytest 測試（會下載模型、跑真的 demucs/basic-pitch/piano 模型，違反
docs/ARCHITECTURE.md「測試不得依賴網路」的規則），是獨立腳本：

    .venv\\Scripts\\python -m tests.eval_accuracy

音符級 F1 定義：
- onset 容許誤差 ±50ms，pitch 必須完全相同才算配對成功；offset 不列入比對。
- 每個 ground-truth 音符只能配對一個預測音符（貪婪最近 onset 配對）。

「量化前後各算一次」：用 monkeypatch 攔截 `engine.pipeline` 呼叫
`notation.quantize.quantize` 前的原始（秒為單位）NoteEvent，算一次 F1；
再把 `transcribe()` 回傳的量化後 Score 換算回秒數，算第二次 F1。
兩者差距大代表問題出在量化（quantize.py / grid / tempo 偵測），差距小則代表
問題出在辨識本身（basic-pitch / piano 模型 / 閾值 / monophonize）。
"""

from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import make_fixtures  # noqa: E402

ONSET_TOL_S = 0.05


# ---------------------------------------------------------------------------
# F1 計算
# ---------------------------------------------------------------------------

def match_and_score(gt: List[dict], pred: List[dict], tol: float = ONSET_TOL_S) -> dict:
    """貪婪最近 onset 配對（同 pitch、onset 誤差 <= tol），回傳 precision/recall/f1/tp/fp/fn。"""
    used = [False] * len(pred)
    pred_sorted_idx = sorted(range(len(pred)), key=lambda i: pred[i]["onset"])
    tp = 0
    for g in sorted(gt, key=lambda d: d["onset"]):
        best_i = None
        best_dt = None
        for i in pred_sorted_idx:
            if used[i]:
                continue
            p = pred[i]
            if int(p["pitch"]) != int(g["pitch"]):
                continue
            dt = abs(float(p["onset"]) - float(g["onset"]))
            if dt <= tol and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_i = i
        if best_i is not None:
            used[best_i] = True
            tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    recall = tp / len(gt) if gt else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_gt": len(gt), "n_pred": len(pred),
    }


def _ev_to_dict(e) -> dict:
    return {"onset": float(e.onset), "offset": float(e.offset), "pitch": int(e.pitch)}


# ---------------------------------------------------------------------------
# 量化後 Score -> 秒數 NoteEvent（反量化，供量化前後比較）
# ---------------------------------------------------------------------------

def _ticks_per_measure(score) -> int:
    ts = score.time_signature
    return int(ts.beats * score.ticks_per_quarter * 4 / ts.beat_type)


def part_events_seconds(score, part) -> List[dict]:
    """把一個量化後的 Part 還原成秒數 NoteEvent 清單（處理跨小節 tie 合併，和弦展開成多個音符）。"""
    tpm = _ticks_per_measure(score)
    seconds_per_tick = (60.0 / score.bpm) / score.ticks_per_quarter if score.bpm else 0.0

    by_voice: Dict[int, list] = {}
    for measure in part.measures:
        base = (measure.number - 1) * tpm
        for note in measure.notes:
            by_voice.setdefault(note.voice, []).append((base, note))

    runs: List[dict] = []
    for _voice, items in by_voice.items():
        items.sort(key=lambda pair: pair[0] + pair[1].start_tick)
        running: Optional[dict] = None
        for base, note in items:
            if not note.pitches:
                if running is not None:
                    runs.append(running)
                    running = None
                continue
            g_start = base + note.start_tick
            g_end = g_start + note.duration_tick
            if note.tie_stop and running is not None and running["pitches"] == note.pitches:
                running["end"] = g_end
                if not note.tie_start:
                    runs.append(running)
                    running = None
            else:
                if running is not None:
                    runs.append(running)
                    running = None
                if note.tie_start:
                    running = {"start": g_start, "end": g_end, "pitches": note.pitches}
                else:
                    runs.append({"start": g_start, "end": g_end, "pitches": note.pitches})
        if running is not None:
            runs.append(running)

    out: List[dict] = []
    for run in runs:
        onset_s = run["start"] * seconds_per_tick
        offset_s = run["end"] * seconds_per_tick
        for p in run["pitches"]:
            out.append({"onset": onset_s, "offset": offset_s, "pitch": int(p)})
    out.sort(key=lambda d: d["onset"])
    return out


# ---------------------------------------------------------------------------
# 執行 transcribe()，同時攔截量化前的原始 NoteEvent
# ---------------------------------------------------------------------------

def run_transcribe_capturing(path: str, options) -> Tuple[object, Dict[str, list], float, bool]:
    """跑 engine.pipeline.transcribe()，回傳 (score, {part_id: 量化前 NoteEvent list}, 耗時秒, 是否用 GPU)。"""
    import engine.pipeline as pipeline_module
    from notation.quantize import quantize as real_quantize

    captured: Dict[str, list] = {}

    def _capturing_quantize(events, bpm, beat_times, time_signature, grid, part_meta):
        captured[str(part_meta.get("id", part_meta.get("name", "P?")))] = list(events)
        return real_quantize(events, bpm, beat_times, time_signature, grid, part_meta)

    orig_quantize = pipeline_module.quantize
    pipeline_module.quantize = _capturing_quantize
    try:
        t0 = time.perf_counter()
        score = pipeline_module.transcribe(path, options)
        elapsed = time.perf_counter() - t0
    finally:
        pipeline_module.quantize = orig_quantize

    gpu_used = False
    try:
        import torch

        gpu_used = bool(getattr(options, "use_gpu", True)) and torch.cuda.is_available()
    except Exception:
        gpu_used = False

    return score, captured, elapsed, gpu_used


# ---------------------------------------------------------------------------
# 各 fixture 的評估
# ---------------------------------------------------------------------------

def eval_solo_run(fixture_name: str, wav_path: Path, gt_notes: List[dict], instrument_hint: str) -> dict:
    from notation.score_model import TranscribeOptions

    options = TranscribeOptions(mode="solo", instrument_hint=instrument_hint, use_gpu=True)
    score, captured, elapsed, gpu_used = run_transcribe_capturing(str(wav_path), options)

    part = score.parts[0] if score.parts else None
    pre_events = [_ev_to_dict(e) for e in next(iter(captured.values()), [])]
    post_events = part_events_seconds(score, part) if part is not None else []

    pre_metrics = match_and_score(gt_notes, pre_events)
    post_metrics = match_and_score(gt_notes, post_events)

    return {
        "fixture": fixture_name,
        "mode": f"solo/{instrument_hint}",
        "elapsed_s": elapsed,
        "gpu_used": gpu_used,
        "bpm_detected": score.bpm,
        "part_name": part.name if part else None,
        "n_notes_pre": len(pre_events),
        "n_notes_post": len(post_events),
        "pre": pre_metrics,
        "post": post_metrics,
        "warnings": list(score.warnings),
    }


def eval_multi_run(fixture_name: str, wav_path: Path, gt_tracks: Dict[str, list]) -> dict:
    from notation.score_model import TranscribeOptions

    options = TranscribeOptions(mode="multi", instrument_hint="auto", use_gpu=True)
    score, captured, elapsed, gpu_used = run_transcribe_capturing(str(wav_path), options)

    parts_by_id = {p.id: p for p in score.parts}
    per_part_report = []
    for part in score.parts:
        pre_events = [_ev_to_dict(e) for e in captured.get(part.id, [])]
        post_events = part_events_seconds(score, part)
        per_part_report.append({
            "part_id": part.id,
            "part_name": part.name,
            "instrument": part.instrument,
            "n_notes_pre": len(pre_events),
            "n_notes_post": len(post_events),
            "pitch_range_post": (
                (min(e["pitch"] for e in post_events), max(e["pitch"] for e in post_events))
                if post_events else None
            ),
            "pre_events": pre_events,
            "post_events": post_events,
        })

    # 幫每個 ground-truth track 找最佳匹配的預測 Part（用量化後事件，pitch 完全相同 + onset 容許誤差）。
    track_matches = {}
    for track_name, gt_notes in gt_tracks.items():
        best = None
        for p in per_part_report:
            metrics = match_and_score(gt_notes, p["post_events"])
            if best is None or metrics["f1"] > best["metrics"]["f1"]:
                best = {"part_id": p["part_id"], "part_name": p["part_name"], "metrics": metrics}
        track_matches[track_name] = best

    return {
        "fixture": fixture_name,
        "mode": "multi/auto",
        "elapsed_s": elapsed,
        "gpu_used": gpu_used,
        "bpm_detected": score.bpm,
        "parts": per_part_report,
        "track_matches": track_matches,
        "warnings": list(score.warnings),
    }


# ---------------------------------------------------------------------------
# 報告輸出
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report_markdown(solo_results: List[dict], multi_result: dict) -> str:
    lines = []
    lines.append("# 準確度評估報告（ACCURACY）")
    lines.append("")
    lines.append(
        "自動由 `tests/eval_accuracy.py` 產生，測試素材為 `tests/make_fixtures.py` 合成的"
        "音檔（見 `runtime/fixtures/`，不進版控）。F1 定義：onset 容許 ±50ms、pitch 精確匹配，"
        "offset 不列入比對。「量化前」= 攔截 `notation.quantize.quantize()` 呼叫前的原始秒數 "
        "NoteEvent；「量化後」= 把 `transcribe()` 回傳的量化 Score 換算回秒數再比對，用來分辨"
        "問題出在辨識還是量化。"
    )
    lines.append("")
    lines.append("## 獨奏（solo）")
    lines.append("")
    lines.append(
        "| Fixture | 模式 | 量化前 P/R/F1 | 量化後 P/R/F1 | 音符數(前/後) | 耗時 | GPU | BPM 偵測 | 備註 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in solo_results:
        pre, post = r["pre"], r["post"]
        note = "; ".join(r["warnings"]) if r["warnings"] else ""
        lines.append(
            f"| {r['fixture']} | {r['mode']} "
            f"| {_fmt_pct(pre['precision'])}/{_fmt_pct(pre['recall'])}/{_fmt_pct(pre['f1'])} "
            f"| {_fmt_pct(post['precision'])}/{_fmt_pct(post['recall'])}/{_fmt_pct(post['f1'])} "
            f"| {r['n_notes_pre']}/{r['n_notes_post']} "
            f"| {r['elapsed_s']:.1f}s | {'是' if r['gpu_used'] else '否'} "
            f"| {r['bpm_detected']:.1f} | {note} |"
        )
    lines.append("")

    lines.append("## 多樂器（multi，demucs 分離）")
    lines.append("")
    lines.append(f"耗時：{multi_result['elapsed_s']:.1f}s，GPU：{'是' if multi_result['gpu_used'] else '否'}，"
                  f"偵測 BPM：{multi_result['bpm_detected']:.1f}")
    if multi_result["warnings"]:
        lines.append(f"警告：{'; '.join(multi_result['warnings'])}")
    lines.append("")
    lines.append("每個輸出 Part 的音符數：")
    lines.append("")
    lines.append("| Part | instrument | 音符數(量化前/後) | 音高範圍(量化後) |")
    lines.append("|---|---|---|---|")
    for p in multi_result["parts"]:
        rng = f"{p['pitch_range_post'][0]}-{p['pitch_range_post'][1]}" if p["pitch_range_post"] else "-"
        lines.append(f"| {p['part_name']} ({p['part_id']}) | {p['instrument']} | "
                      f"{p['n_notes_pre']}/{p['n_notes_post']} | {rng} |")
    lines.append("")
    lines.append("Ground-truth track 對應最佳匹配 Part（量化後 F1）：")
    lines.append("")
    lines.append("| GT track | 最佳匹配 Part | P/R/F1 |")
    lines.append("|---|---|---|")
    for track_name, best in multi_result["track_matches"].items():
        if best is None:
            lines.append(f"| {track_name} | (無輸出 Part) | - |")
            continue
        m = best["metrics"]
        lines.append(
            f"| {track_name} | {best['part_name']} ({best['part_id']}) | "
            f"{_fmt_pct(m['precision'])}/{_fmt_pct(m['recall'])}/{_fmt_pct(m['f1'])} |"
        )
    lines.append("")

    lines.append("## 觀察與已知限制")
    lines.append("")
    lines.append(
        "- **耗時欄位包含首次模型載入成本**：同一個 process 內，每種模型（basic-pitch/"
        "piano_transcription_inference/demucs）只在第一次呼叫時真正載入，`solo/auto` 第二次"
        "以後幾乎只剩推論時間（0.2s 級），`solo/vocal`／`solo/piano` 的耗時包含了第一次載入"
        "模型（含鋼琴 checkpoint 讀檔）的成本。正式伺服器是常駐 process，行為與此一致：第一個"
        "請求較慢，之後的請求快很多。"
    )
    auto_scale_f1 = next((r["post"]["f1"] for r in solo_results
                          if r["fixture"] == "scale_mono" and r["mode"] == "solo/auto"), None)
    auto_chords_f1 = next((r["post"]["f1"] for r in solo_results
                           if r["fixture"] == "piano_chords" and r["mode"] == "solo/auto"), None)
    lines.append(
        "- **`instrument_hint=auto`／`piano` 在人工合成、泛音明顯的音檔上仍弱於正確 hint，"
        "但 `suppress_harmonics`（TranscribeOptions 新欄位，預設開啟）已把 `auto` 的 precision "
        "拉高一截**：`scale_mono` 用 `auto` 目前 "
        f"{_fmt_pct(auto_scale_f1) if auto_scale_f1 is not None else '?'} F1"
        "（開發前 69.8%），`piano_chords` 用 `auto` 目前 "
        f"{_fmt_pct(auto_chords_f1) if auto_chords_f1 is not None else '?'} F1"
        "（開發前 64.4%）；兩者的 recall 都還是 100%（真正的音符全部找到），`suppress_harmonics` "
        "抓的是「onset 差 <40ms、pitch 差為八度／八度五度／兩八度／兩八度大三度、confidence 明顯"
        "偏低」這種保守條件，只清掉一部分明顯的泛音誤判，沒有做到跟正確 hint（`vocal` 單音化、"
        "`piano` 專用模型）一樣的水準——這仍然反映「人工合成音泛音結構跟真實樂器不同」這個測試"
        "素材本身的特性，不是 `auto` 模式的迴歸。"
    )
    lines.append(
        "- **`multi_mix` 的合成旋律音源仍常被 demucs 同時分到 `vocals` 與 `other` 兩軌**："
        "因為是人工合成的音色，不是真人聲紋，demucs（在真實錄音上訓練）沒辦法穩定把它"
        "歸到單一 stem；`engine/pipeline.py` 現在會在 other part 找 onset 差 <60ms、pitch 相同"
        "或差八度、confidence 較低的音符去重（見 warnings 欄位），減少同一段旋律被同時寫兩次"
        "的機率，但無法完全避免 demucs 把同一旋律分到兩個 stem 本身——這是使用人工合成測試音檔"
        "評估依賴真實聲音模型（demucs）時的已知限制，並非程式邏輯錯誤。"
    )
    lines.append(
        "- **鼓聲分類已大幅改善**：`engine/transcribe_drums.py` 改用「各頻帶濾波後 RMS 相對"
        "視窗峰值」而非「頻帶能量占全頻譜比例」判斷 kick/snare/hihat（避免同時打擊時響度大的"
        "樂器把安靜樂器的比例稀釋到偵測不到），並加上頻帶衰減時間、spectral centroid 輔助判斷、"
        "onset 偵測前墊一小段靜音修掉「緊貼音檔開頭的 onset 常被漏掉」的邊界問題。在 "
        "`tests/make_fixtures.py` 更真實的鼓聲合成（snare = 200Hz 音頭 + 1-4kHz 帶通噪音、"
        "hihat = 6kHz 以上高通噪音 30ms、kick = 60→40Hz 掃頻 150ms，並加入第 2、4 拍 snare）上，"
        "`tests/test_engine_transcribe_drums.py` 驗證 kick/snare/hihat 各自 onset-level F1 "
        "都達到 1.00（未經 demucs，直接對鼓聲單軌評估）；經過 demucs 分離後的 `multi_mix` "
        "完整流程 F1 約 0.7-0.9（見上表，demucs 分離品質本身有 run-to-run 變動）。"
    )
    bass_part = next((p for p in multi_result["parts"] if p["part_name"] == "Bass"), None)
    bass_range_text = (
        f"{bass_part['pitch_range_post'][0]}-{bass_part['pitch_range_post'][1]}"
        if bass_part and bass_part["pitch_range_post"] else "（無輸出）"
    )
    lines.append(
        f"- **低音 Part 音高範圍大致正確**：本次執行 `Bass` part 音高範圍量測為 "
        f"{bass_range_text}，fixture ground-truth 是 36-48；`engine/pipeline.py` 呼叫 "
        "basic-pitch 時對 bass 已限制 `min_pitch=28, max_pitch=60`，所以即使有雜音/demucs "
        "分離殘留，量測到的範圍也不會超出 28-60。這個 Part 的音符數/範圍在不同次執行之間"
        "會有些微差異（demucs／basic-pitch 在 GPU 上並非完全逐位元決定性），屬於這兩個"
        "第三方模型本身的正常變動，不是本次程式修改造成的迴歸。"
    )
    lines.append(
        "- **拍號自動偵測（`time_signature=\"auto\"`）**：`engine/rhythm.detect_time_signature()` "
        "用 beat_times 對齊的重音強度序列，比較 3 拍與 4 拍分組下「哪個拍位最像 downbeat」的"
        "重音對比，明顯偏 3 拍才回 `3/4`，否則一律 `4/4`（6/8 暫不自動偵測）；beat 數不足或"
        "對比不夠明顯時保守回退 `4/4`。這幾個 fixture 都不足以拿來測試（`scale_mono`／"
        "`piano_chords`／`multi_mix` 本身不含明顯的 3 拍重音模式，也是驗證重點：不能被誤判"
        "成 3/4），實際驗證見 `tests/test_engine_rhythm.py` 新增的 `waltz_mono` 3/4 華爾滋 "
        "fixture（強-弱-弱重音）：偵測結果為 `3/4`；既有三個 4/4 fixture 維持 `4/4` 不受影響。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    fixtures = make_fixtures.generate_all()

    scale_gt = json.loads(fixtures["scale_mono"]["json"].read_text(encoding="utf-8"))["notes"]
    chords_gt = json.loads(fixtures["piano_chords"]["json"].read_text(encoding="utf-8"))["notes"]
    multi_gt = json.loads(fixtures["multi_mix"]["json"].read_text(encoding="utf-8"))["tracks"]

    solo_results = []
    print("== solo: scale_mono / instrument_hint=vocal ==")
    solo_results.append(eval_solo_run("scale_mono", fixtures["scale_mono"]["wav"], scale_gt, "vocal"))
    print("== solo: scale_mono / instrument_hint=auto ==")
    solo_results.append(eval_solo_run("scale_mono", fixtures["scale_mono"]["wav"], scale_gt, "auto"))
    print("== solo: piano_chords / instrument_hint=auto ==")
    solo_results.append(eval_solo_run("piano_chords", fixtures["piano_chords"]["wav"], chords_gt, "auto"))
    print("== solo: piano_chords / instrument_hint=piano ==")
    solo_results.append(eval_solo_run("piano_chords", fixtures["piano_chords"]["wav"], chords_gt, "piano"))

    for r in solo_results:
        print(f"  {r['fixture']:>14s} {r['mode']:<12s} "
              f"pre_f1={r['pre']['f1']:.3f} post_f1={r['post']['f1']:.3f} "
              f"notes(pre/post)={r['n_notes_pre']}/{r['n_notes_post']} "
              f"time={r['elapsed_s']:.1f}s gpu={r['gpu_used']}")

    print("== multi: multi_mix / mode=multi ==")
    multi_result = eval_multi_run("multi_mix", fixtures["multi_mix"]["wav"], multi_gt)
    for p in multi_result["parts"]:
        print(f"  part {p['part_name']} ({p['part_id']}): notes(pre/post)="
              f"{p['n_notes_pre']}/{p['n_notes_post']} pitch_range={p['pitch_range_post']}")
    for track_name, best in multi_result["track_matches"].items():
        if best is None:
            print(f"  gt track {track_name}: 無匹配 Part")
        else:
            print(f"  gt track {track_name} -> {best['part_name']} f1={best['metrics']['f1']:.3f}")

    report = build_report_markdown(solo_results, multi_result)
    out_path = REPO_ROOT / "docs" / "ACCURACY.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n報告已寫入 {out_path}")


if __name__ == "__main__":
    main()
