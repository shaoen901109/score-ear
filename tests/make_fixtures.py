# -*- coding: utf-8 -*-
"""合成測試用音訊 fixtures（供 tests/eval_accuracy.py 與人工試聽使用）。

輸出到 `runtime/fixtures/`（不進版控，`.gitignore` 已排除 `*.wav`；JSON 的
ground-truth 檔案也一併放在這裡，方便跟對應的 wav 放在一起管理，不需要進版控）。

三個 fixture：
(a) scale_mono      單音旋律：C 大調音階上下行，每音 0.4 秒、120 BPM，
                    加法合成（近似鋸齒波，含泛音）+ ADSR 包絡。
(b) piano_chords    鋼琴式和弦進行 C-F-G-C，每個和弦 1 秒：右手為持續的
                    block chord，左手疊加一段分解和弦（arpeggio）。
(c) multi_mix       多樂器混音：高音旋律 + 低音線（MIDI 36-48，每拍一音）
                    + 鼓（kick 每拍、hihat 每半拍）。

每個 fixture 都輸出對應的 ground-truth NoteEvent JSON（單位：秒），
給 `tests/eval_accuracy.py` 算音符級 F1 用。

這個檔案故意不用 `test_` 開頭，避免被 pytest 收集成測試案例；用
`.venv\\Scripts\\python -m tests.make_fixtures` 或直接 `python tests/make_fixtures.py`
執行即可重新產生 fixtures。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "runtime" / "fixtures"

SR = 22050
BPM = 120.0
BEAT_S = 60.0 / BPM  # 0.5s at 120 BPM

# GM 打擊音號（與 docs/ARCHITECTURE.md 一致）
GM_KICK = 36
GM_SNARE = 38
GM_HIHAT = 42


# ---------------------------------------------------------------------------
# 基礎工具
# ---------------------------------------------------------------------------

def midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def adsr_envelope(
    n_samples: int,
    sr: int,
    attack: float = 0.015,
    decay: float = 0.06,
    sustain_level: float = 0.65,
    release: float = 0.05,
) -> np.ndarray:
    """簡單線性 ADSR 包絡，長度為 n_samples。"""
    a = max(1, int(sr * attack))
    d = max(1, int(sr * decay))
    r = max(1, int(sr * release))
    sustain_len = max(0, n_samples - a - d - r)

    env = np.concatenate([
        np.linspace(0.0, 1.0, a, endpoint=False),
        np.linspace(1.0, sustain_level, d, endpoint=False),
        np.full(sustain_len, sustain_level, dtype=np.float64),
        np.linspace(sustain_level, 0.0, r, endpoint=True),
    ])
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)))
    return env[:n_samples].astype(np.float32)


def harmonic_tone(
    freq: float,
    duration_s: float,
    sr: int,
    n_harmonics: int = 8,
    amplitude: float = 0.3,
    use_adsr: bool = True,
) -> np.ndarray:
    """加法合成近似鋸齒波（含泛音）：sum (-1)^(k+1)/k * sin(2*pi*k*f*t)，可選 ADSR。"""
    n = int(round(sr * duration_s))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    wave = np.zeros(n, dtype=np.float64)
    nyquist = sr / 2.0
    for k in range(1, n_harmonics + 1):
        hfreq = freq * k
        if hfreq >= nyquist:
            break
        sign = 1.0 if (k % 2 == 1) else -1.0
        wave += (sign / k) * np.sin(2 * np.pi * hfreq * t)
    wave *= 2.0 / np.pi
    wave *= amplitude
    if use_adsr:
        wave = wave * adsr_envelope(n, sr)
    return wave.astype(np.float32)


def piano_like_tone(
    freq: float,
    duration_s: float,
    sr: int,
    n_harmonics: int = 10,
    amplitude: float = 0.35,
    attack_s: float = 0.004,
) -> np.ndarray:
    """近似鋼琴音色：快速 attack + 每個泛音各自指數衰減（高次泛音衰減更快）。

    跟 `harmonic_tone()`（ADSR、長時間維持 sustain）不同：真實鋼琴／弦樂器
    沒有平坦的 sustain 段，音量從觸鍵後就持續衰減，且高次泛音衰減得比基音快
    （聽起來「發亮」的高頻部分很快消失，只留下較悶的低次泛音）。這個差異
    對辨識準確度不是裝飾用：`piano_transcription_inference` 是在真實鋼琴錄音
    上訓練的 CNN，對缺乏真實觸鍵起音（transient）與泛音衰減特性的純加法波形
    （尤其是長時間等幅的 sustain）辨識效果明顯較差，改成這種包絡後 onset
    偵測穩定許多。
    """
    n = int(round(sr * duration_s))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    wave = np.zeros(n, dtype=np.float64)
    nyquist = sr / 2.0
    for k in range(1, n_harmonics + 1):
        hfreq = freq * k
        if hfreq >= nyquist:
            break
        decay_rate = 2.0 + 1.1 * k  # 每秒衰減率；高次泛音衰減更快
        wave += (1.0 / k) * np.exp(-t * decay_rate) * np.sin(2 * np.pi * hfreq * t)
    wave *= amplitude

    a = max(1, int(sr * attack_s))
    env = np.ones(n, dtype=np.float64)
    env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    wave *= env
    return wave.astype(np.float32)


def _fft_bandpass_noise(n: int, sr: int, lo: float, hi: float, seed: int) -> np.ndarray:
    """用 FFT 遮罩把白噪音濾成 [lo, hi) Hz 的帶通噪音（lo=0 即高通）。

    比起用 scipy.signal 設計 IIR/FIR 濾波器，FFT 遮罩沒有濾波器暫態響應與
    相位失真問題，對「產生測試素材」這種一次性合成用途更單純、更可預期。
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float64)
    if n <= 1:
        return noise.astype(np.float32)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    spectrum = spectrum * mask
    filtered = np.fft.irfft(spectrum, n=n)
    peak = np.max(np.abs(filtered)) or 1.0
    return (filtered / peak).astype(np.float32)


def kick_hit(duration_s: float = 0.15, sr: int = 22050, amplitude: float = 0.9) -> np.ndarray:
    """60Hz → 40Hz 線性掃頻 + 快速指數衰減，模擬大鼓 (kick)，預設 150ms。"""
    n = int(round(sr * duration_s))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    freq_env = 60.0 + (40.0 - 60.0) * (t / duration_s)  # 60Hz 線性滑到 40Hz
    phase = 2 * np.pi * np.cumsum(freq_env) / sr
    amp_env = np.exp(-t * 14.0)
    wave = amplitude * amp_env * np.sin(phase)
    return wave.astype(np.float32)


def snare_hit(duration_s: float = 0.12, sr: int = 22050, amplitude: float = 0.8, seed: int = 0) -> np.ndarray:
    """200Hz 正弦（音頭）+ 1-4kHz 帶通噪音（噪聲毯），模擬 snare。

    真實小鼓聲音是鼓皮的音高成分（低頻正弦/準正弦）疊加響弦的中高頻噪音，
    衰減比 kick 快、比 hihat 慢，介於兩者之間。
    """
    n = int(round(sr * duration_s))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    tone_env = np.exp(-t * 25.0)
    tone = np.sin(2 * np.pi * 200.0 * t) * tone_env

    noise = _fft_bandpass_noise(n, sr, 1000.0, 4000.0, seed=seed).astype(np.float64)
    noise_env = np.exp(-t * 18.0)
    noise_part = noise * noise_env

    wave = 0.45 * tone + 0.75 * noise_part
    peak = np.max(np.abs(wave)) or 1.0
    wave = wave / peak * amplitude
    return wave.astype(np.float32)


def hihat_hit(duration_s: float = 0.03, sr: int = 22050, amplitude: float = 0.35, seed: int = 0) -> np.ndarray:
    """6kHz 以上高通噪音，30ms，快速指數衰減，模擬閉合 hihat。"""
    n = int(round(sr * duration_s))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    filtered = _fft_bandpass_noise(n, sr, 6000.0, sr / 2.0, seed=seed).astype(np.float64)
    t = np.arange(n, dtype=np.float64) / sr
    amp_env = np.exp(-t * 90.0)
    wave = amplitude * amp_env * filtered
    peak = np.max(np.abs(wave)) or 1.0
    wave = wave / peak * amplitude
    return wave.astype(np.float32)


def place(canvas: np.ndarray, sound: np.ndarray, onset_s: float, sr: int) -> None:
    """把 sound 疊加進 canvas 的 onset_s 秒位置（就地修改，超出長度自動截斷）。"""
    start = int(round(onset_s * sr))
    end = start + len(sound)
    if start >= len(canvas):
        return
    end = min(end, len(canvas))
    canvas[start:end] += sound[: end - start]


def normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    if m <= 1e-9:
        return y
    return (y * (peak / m)).astype(np.float32)


def _note_dict(onset: float, offset: float, pitch: int, velocity: int = 90, confidence: float = 1.0) -> dict:
    return {
        "onset": round(float(onset), 6),
        "offset": round(float(offset), 6),
        "pitch": int(pitch),
        "velocity": int(velocity),
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------------------
# (a) 單音旋律：C 大調音階上下行
# ---------------------------------------------------------------------------

def generate_scale_mono():
    note_dur = 0.4
    midis = [60, 62, 64, 65, 67, 69, 71, 72,  # 上行 C4..C5
             71, 69, 67, 65, 64, 62, 60]      # 下行 B4..C4（不重複頂音）

    total_s = note_dur * len(midis)
    canvas = np.zeros(int(math.ceil(total_s * SR)) + SR, dtype=np.float32)
    notes: List[dict] = []
    for i, m in enumerate(midis):
        onset = i * note_dur
        tone = harmonic_tone(midi_to_freq(m), note_dur * 0.92, SR, n_harmonics=10, amplitude=0.35)
        place(canvas, tone, onset, SR)
        notes.append(_note_dict(onset, onset + note_dur * 0.92, m))

    canvas = canvas[: int(round(total_s * SR))]
    canvas = normalize(canvas)
    return canvas, notes, total_s


# ---------------------------------------------------------------------------
# (b) 鋼琴式和弦進行 C-F-G-C，附加分解和弦
# ---------------------------------------------------------------------------

def generate_piano_chords():
    chord_dur = 1.0
    progression = [
        {"block": [60, 64, 67], "arp": [48, 52, 55, 60]},   # C
        {"block": [65, 69, 72], "arp": [53, 57, 60, 65]},   # F
        {"block": [67, 71, 74], "arp": [55, 59, 62, 67]},   # G
        {"block": [60, 64, 67], "arp": [48, 52, 55, 60]},   # C
    ]
    total_s = chord_dur * len(progression)
    canvas = np.zeros(int(math.ceil(total_s * SR)) + SR, dtype=np.float32)
    notes: List[dict] = []

    for i, chord in enumerate(progression):
        chord_onset = i * chord_dur
        # 右手：block chord，整個 1 秒持續（鋼琴式：音量從觸鍵後就開始衰減，
        # 不是平的 sustain，見 piano_like_tone() 的說明）。
        for pitch in chord["block"]:
            tone = piano_like_tone(midi_to_freq(pitch), chord_dur * 0.98, SR, n_harmonics=10, amplitude=0.3)
            place(canvas, tone, chord_onset, SR)
            notes.append(_note_dict(chord_onset, chord_onset + chord_dur * 0.98, pitch))

        # 左手：分解和弦，1 秒內 4 個十六分音符（0.25s 一個）。
        arp_note_dur = chord_dur / len(chord["arp"])
        for j, pitch in enumerate(chord["arp"]):
            onset = chord_onset + j * arp_note_dur
            tone = piano_like_tone(midi_to_freq(pitch), arp_note_dur * 0.95, SR, n_harmonics=10, amplitude=0.32)
            place(canvas, tone, onset, SR)
            notes.append(_note_dict(onset, onset + arp_note_dur * 0.95, pitch))

    canvas = canvas[: int(round(total_s * SR))]
    canvas = normalize(canvas)
    notes.sort(key=lambda d: (d["onset"], d["pitch"]))
    return canvas, notes, total_s


# ---------------------------------------------------------------------------
# (b2) 3/4 華爾滋單音旋律：強-弱-弱重音，供 engine.rhythm.detect_time_signature 測試
# ---------------------------------------------------------------------------

WALTZ_BPM = 90.0


def generate_waltz_mono():
    """3/4 拍單音旋律：每小節 3 拍，第 1 拍（強拍）音量大、時值長；
    第 2、3 拍（弱拍）音量小、時值短，重複 8 小節，製造明顯的「強-弱-弱」重音模式。
    """
    beat_s = 60.0 / WALTZ_BPM
    n_measures = 8
    pattern_midis = [60, 64, 67]  # 每小節：C4(強) E4(弱) G4(弱)

    total_s = n_measures * 3 * beat_s
    canvas = np.zeros(int(math.ceil(total_s * SR)) + SR, dtype=np.float32)
    notes: List[dict] = []

    for m in range(n_measures):
        for beat_in_measure in range(3):
            onset = (m * 3 + beat_in_measure) * beat_s
            pitch = pattern_midis[beat_in_measure]
            if beat_in_measure == 0:
                amplitude, dur = 0.42, beat_s * 0.92
            else:
                amplitude, dur = 0.14, beat_s * 0.55
            tone = harmonic_tone(midi_to_freq(pitch), dur, SR, n_harmonics=8, amplitude=amplitude)
            place(canvas, tone, onset, SR)
            velocity = 110 if beat_in_measure == 0 else 55
            notes.append(_note_dict(onset, onset + dur, pitch, velocity=velocity))

    canvas = canvas[: int(round(total_s * SR))]
    canvas = normalize(canvas)
    return canvas, notes, total_s


# ---------------------------------------------------------------------------
# (c) 多樂器混音：旋律 + 低音線 + 鼓
# ---------------------------------------------------------------------------

def generate_multi_mix():
    n_beats = 8
    total_s = n_beats * BEAT_S

    melody_midis = [72, 74, 76, 77, 79, 77, 76, 72]     # C5 D5 E5 F5 G5 F5 E5 C5，每拍一音
    bass_midis = [36, 43, 45, 41, 36, 40, 43, 36]        # 限制在 MIDI 36-48
    # 8 拍視為兩小節 4/4：backbeat 在每小節第 2、4 拍（0-indexed 1,3,5,7）。
    snare_beats = {1, 3, 5, 7}

    canvas = np.zeros(int(math.ceil(total_s * SR)) + SR, dtype=np.float32)
    drums_canvas = np.zeros_like(canvas)

    melody_notes: List[dict] = []
    bass_notes: List[dict] = []
    kick_notes: List[dict] = []
    snare_notes: List[dict] = []
    hihat_notes: List[dict] = []

    for i in range(n_beats):
        onset = i * BEAT_S

        # 旋律（高音）
        m_pitch = melody_midis[i]
        m_tone = harmonic_tone(midi_to_freq(m_pitch), BEAT_S * 0.85, SR, n_harmonics=10, amplitude=0.3)
        place(canvas, m_tone, onset, SR)
        melody_notes.append(_note_dict(onset, onset + BEAT_S * 0.85, m_pitch))

        # 低音線（每拍一音）
        b_pitch = bass_midis[i]
        b_tone = harmonic_tone(midi_to_freq(b_pitch), BEAT_S * 0.9, SR, n_harmonics=6, amplitude=0.32)
        place(canvas, b_tone, onset, SR)
        bass_notes.append(_note_dict(onset, onset + BEAT_S * 0.9, b_pitch))

        # 鼓：kick 每拍一次（第 2、4 拍改由 snare 主導，kick 不與 snare 同時打）
        if i not in snare_beats:
            kick = kick_hit(0.15, SR, amplitude=0.85)
            place(canvas, kick, onset, SR)
            place(drums_canvas, kick, onset, SR)
            kick_notes.append(_note_dict(onset, onset + 0.15, GM_KICK, velocity=110))
        else:
            snare = snare_hit(0.12, SR, amplitude=0.8, seed=i)
            place(canvas, snare, onset, SR)
            place(drums_canvas, snare, onset, SR)
            snare_notes.append(_note_dict(onset, onset + 0.12, GM_SNARE, velocity=105))

        # 鼓：hihat 每半拍一次（本拍正拍 + 半拍後）
        for half in (0.0, 0.5):
            hh_onset = onset + half * BEAT_S
            hh = hihat_hit(0.03, SR, amplitude=0.3, seed=i * 2 + int(half))
            place(canvas, hh, hh_onset, SR)
            place(drums_canvas, hh, hh_onset, SR)
            hihat_notes.append(_note_dict(hh_onset, hh_onset + 0.03, GM_HIHAT, velocity=70))

    canvas = canvas[: int(round(total_s * SR))]
    canvas = normalize(canvas)
    drums_canvas = drums_canvas[: int(round(total_s * SR))]
    drums_canvas = normalize(drums_canvas)

    tracks = {
        "melody": melody_notes,
        "bass": bass_notes,
        "drums": sorted(kick_notes + snare_notes + hihat_notes, key=lambda d: d["onset"]),
    }
    return canvas, drums_canvas, tracks, total_s


# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------

FIXTURE_SPECS = {
    "scale_mono": {"wav": "scale_mono.wav", "json": "scale_mono.notes.json", "kind": "solo"},
    "piano_chords": {"wav": "piano_chords.wav", "json": "piano_chords.notes.json", "kind": "solo"},
    "waltz_mono": {"wav": "waltz_mono.wav", "json": "waltz_mono.notes.json", "kind": "solo"},
    "multi_mix": {"wav": "multi_mix.wav", "json": "multi_mix.notes.json", "kind": "multi"},
    # 鼓聲單軌（多樂器混音分離前的原始鼓聲，未經 demucs），供鼓分類器直接調參與評估用，
    # 不必依賴 demucs 分離品質即可衡量 engine/transcribe_drums.py 本身的鼓聲分類準確度。
    "multi_mix_drums": {"wav": "multi_mix.drums_only.wav", "json": "multi_mix.notes.json", "kind": "drums_only"},
}


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_all(force: bool = False) -> dict:
    """產生全部 fixtures，回傳 {name: {wav: Path, json: Path}}。已存在且非 force 時跳過重算。"""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    result = {}

    wav_path = FIXTURES_DIR / FIXTURE_SPECS["scale_mono"]["wav"]
    json_path = FIXTURES_DIR / FIXTURE_SPECS["scale_mono"]["json"]
    if force or not wav_path.exists() or not json_path.exists():
        y, notes, total_s = generate_scale_mono()
        _write_wav(wav_path, y, SR)
        _write_json(json_path, {
            "fixture": "scale_mono", "sr": SR, "bpm": BPM, "duration_s": total_s, "notes": notes,
        })
    result["scale_mono"] = {"wav": wav_path, "json": json_path}

    wav_path = FIXTURES_DIR / FIXTURE_SPECS["piano_chords"]["wav"]
    json_path = FIXTURES_DIR / FIXTURE_SPECS["piano_chords"]["json"]
    if force or not wav_path.exists() or not json_path.exists():
        y, notes, total_s = generate_piano_chords()
        _write_wav(wav_path, y, SR)
        _write_json(json_path, {
            "fixture": "piano_chords", "sr": SR, "bpm": BPM, "duration_s": total_s, "notes": notes,
        })
    result["piano_chords"] = {"wav": wav_path, "json": json_path}

    wav_path = FIXTURES_DIR / FIXTURE_SPECS["waltz_mono"]["wav"]
    json_path = FIXTURES_DIR / FIXTURE_SPECS["waltz_mono"]["json"]
    if force or not wav_path.exists() or not json_path.exists():
        y, notes, total_s = generate_waltz_mono()
        _write_wav(wav_path, y, SR)
        _write_json(json_path, {
            "fixture": "waltz_mono", "sr": SR, "bpm": WALTZ_BPM, "duration_s": total_s, "notes": notes,
        })
    result["waltz_mono"] = {"wav": wav_path, "json": json_path}

    wav_path = FIXTURES_DIR / FIXTURE_SPECS["multi_mix"]["wav"]
    json_path = FIXTURES_DIR / FIXTURE_SPECS["multi_mix"]["json"]
    drums_wav_path = FIXTURES_DIR / FIXTURE_SPECS["multi_mix_drums"]["wav"]
    if force or not wav_path.exists() or not json_path.exists() or not drums_wav_path.exists():
        y, drums_y, tracks, total_s = generate_multi_mix()
        _write_wav(wav_path, y, SR)
        _write_wav(drums_wav_path, drums_y, SR)
        _write_json(json_path, {
            "fixture": "multi_mix", "sr": SR, "bpm": BPM, "duration_s": total_s, "tracks": tracks,
        })
    result["multi_mix"] = {"wav": wav_path, "json": json_path}
    result["multi_mix_drums"] = {"wav": drums_wav_path, "json": json_path}

    return result


def main() -> None:
    result = generate_all(force=True)
    for name, paths in result.items():
        print(f"{name}: {paths['wav']}  {paths['json']}")


if __name__ == "__main__":
    main()
