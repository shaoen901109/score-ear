# 聽歌寫譜（Windows 免安裝版）架構與模組契約

目標：上傳錄音 → 辨識音符 → 輸出五線譜與簡譜。獨奏（鋼琴、人聲）要穩定；多樂器要能先分離音源再各自寫譜。
執行形態：Python 3.10 + PyInstaller onedir 打包成資料夾，雙擊 `聽歌寫譜.exe` 啟動本地 FastAPI 伺服器並開瀏覽器。無需安裝。

## 套件路線
- 解碼：soundfile（wav/flac/ogg）；mp3/m4a/mp4 走 imageio-ffmpeg 附帶的 ffmpeg 轉 wav。
- 音源分離：demucs `htdemucs`（vocals / drums / bass / other）。GPU 有就用，沒有就 CPU。
- 鋼琴：piano_transcription_inference（ByteDance）。
- 通用多音（other、bass、獨奏未知樂器）：basic-pitch。
- 人聲／單音旋律：basic-pitch 結果做單音化後處理（同一時刻只留能量最高音）；備援 librosa.pyin。
- 鼓：onset 偵測 + 頻帶能量分類（kick / snare / hihat），寫成打擊譜（單線譜）。
- 節拍與速度：librosa.beat.beat_track + onset 強度；使用者可手動覆寫 BPM。
- 調性：Krumhansl-Schmuckler 音級輪廓相關（自製，不依賴外部）。
- 五線譜渲染：verovio（MusicXML → SVG，伺服器端）。
- 簡譜渲染：自製 SVG（`notation/jianpu.py`）。
- MusicXML 與 MIDI 匯出：自製寫入器（不依賴 music21，減少打包體積與啟動時間）。

## 目錄
```
聽歌寫譜/
  app/
    __init__.py
    main.py              # 入口：啟動 uvicorn、開瀏覽器；PyInstaller 打包目標
    server.py            # FastAPI：上傳、任務佇列、進度、結果、下載
    static/              # 前端 HTML/JS/CSS（純靜態，不用打包工具）
  engine/
    __init__.py
    audio_io.py          # load_audio(path) -> (mono float32 np.ndarray, sr=22050 或原 sr)；ffmpeg 轉檔
    separate.py          # separate(y, sr) -> dict[stem_name, np.ndarray]
    transcribe_piano.py  # -> list[NoteEvent]
    transcribe_poly.py   # basic-pitch -> list[NoteEvent]
    transcribe_mono.py   # 單音化 / pyin -> list[NoteEvent]
    transcribe_drums.py  # -> list[NoteEvent]（pitch 用 GM 打擊音號：36 kick, 38 snare, 42 hihat）
    rhythm.py            # detect_tempo_and_beats(y, sr) -> (bpm, beat_times)
    key.py               # detect_key(note_events) -> KeySignature
    pipeline.py          # transcribe(path, options, progress) -> Score  ← 對外唯一入口
  notation/
    __init__.py
    score_model.py       # 資料模型（下方定義），to_dict/from_dict
    quantize.py          # NoteEvent(秒) -> Score(拍格)
    musicxml.py          # score_to_musicxml(score) -> str
    midi.py              # score_to_midi_bytes(score) -> bytes
    jianpu.py            # score_to_jianpu_svg(score) -> str (SVG)
    staff.py             # score_to_staff_svg(score) -> list[str]（每頁一個 SVG，內部呼叫 verovio）
  tests/
  build/build.ps1        # PyInstaller 打包腳本
  docs/
```

## 共用資料模型（notation/score_model.py，全部 dataclass，可 JSON 序列化）

```python
@dataclass
class NoteEvent:            # 辨識階段的原始音符（單位：秒）
    onset: float            # 起始秒
    offset: float           # 結束秒（> onset）
    pitch: int              # MIDI 音高 0-127；鼓用 GM 打擊音號
    velocity: int = 80      # 1-127
    confidence: float = 1.0 # 0-1

@dataclass
class KeySignature:
    fifths: int             # -7..7（負為降記號）
    mode: str               # "major" | "minor"
    tonic: str              # 例 "C", "F#", "Bb"

@dataclass
class TimeSignature:
    beats: int              # 4
    beat_type: int          # 4

@dataclass
class Note:                 # 量化後音符（單位：拍格 tick，每四分音符 = 480 tick）
    start_tick: int         # 相對小節開頭
    duration_tick: int      # > 0
    pitches: list[int]      # 空 list = 休止符；多個 = 和弦
    tie_start: bool = False # 跨小節連結線
    tie_stop: bool = False
    voice: int = 1

@dataclass
class Measure:
    number: int             # 從 1 開始
    notes: list[Note]       # 依 start_tick 排序，同 voice 內不重疊，總長 = 小節 tick 數（休止符補滿）

@dataclass
class Part:
    id: str                 # "P1"...
    name: str               # "Piano", "Vocal", "Bass", "Drums", "Other"
    instrument: str         # "piano" | "vocal" | "bass" | "drums" | "guitar" | "other"
    clef: str               # "treble" | "bass" | "percussion" | "grand"（grand = 鋼琴雙譜表，量化時 pitch < 60 放低音譜）
    midi_program: int       # GM program 0-127；鼓忽略
    measures: list[Measure]

@dataclass
class Score:
    title: str
    bpm: float
    time_signature: TimeSignature
    key: KeySignature
    ticks_per_quarter: int = 480
    parts: list[Part]
    source_file: str = ""
    warnings: list[str]     # 例如「速度偵測信心低」
```

## 對外函式契約
- `engine.pipeline.transcribe(path: str, options: TranscribeOptions, progress: Callable[[float, str], None] | None) -> Score`
  - `TranscribeOptions`: `mode: "solo" | "multi"`, `instrument_hint: "auto"|"piano"|"vocal"|"guitar"|"other"`（solo 用）, `bpm_override: float | None`, `time_signature: "auto"|"4/4"|"3/4"|"6/8"`, `min_note_ms: int = 60`, `quantize: "1/16"|"1/8"|"1/32"`(最小格), `use_gpu: bool = True`
  - progress(fraction 0-1, 中文訊息)。
  - solo 模式：不分離；依 hint 選轉譜器（auto = basic-pitch，若 hint=piano 用鋼琴模型，vocal 用單音）。
  - multi 模式：demucs 分離 → vocals→單音、drums→鼓、bass→basic-pitch(限 MIDI 28-60)、other→先用鋼琴模型與 basic-pitch 擇一（預設 basic-pitch）。全空的 stem 不建 Part。
- `notation.quantize.quantize(events, bpm, beat_times, time_signature, grid, part_meta) -> Part`
- `notation.musicxml.score_to_musicxml(score) -> str`（MusicXML 3.1 partwise，含 divisions、key、time、clef、tie、和弦 `<chord/>`、休止符、鼓用 unpitched）
- `notation.midi.score_to_midi_bytes(score) -> bytes`
- `notation.jianpu.score_to_jianpu_svg(score, part_index: int | None = None) -> str`
  - 簡譜規則：1-7 為調內級數（以 Score.key.tonic 為 1；minor 以其平行大調的 1 表示並標「1=X」），高八度上加點、低八度下加點，休止 0，附點「.」，時值下劃線（1 條=八分、2 條=十六分），二分音符後加「-」延長線，升降用 # b 前綴；每行 4 小節，小節線「|」，開頭寫「1=C 4/4 ♩=120」。和弦垂直堆疊。
- `notation.staff.score_to_staff_svg(score) -> list[str]`（verovio，頁面 A4 直式，scale 40）
- `app.server`：
  - `POST /api/jobs`（multipart: file, options JSON）→ `{job_id}`
  - `GET /api/jobs/{id}` → `{status: queued|running|done|error, progress, message, error}`
  - `GET /api/jobs/{id}/result` → `{score: Score.to_dict(), staff_svg: [...], jianpu_svg: "..."}`
  - `GET /api/jobs/{id}/download/{fmt}`，fmt ∈ musicxml | midi | jianpu.svg | staff.svg（zip 多頁）| json
  - 任務單一 worker 執行緒依序處理；模型延遲載入並快取。

## 規範
- 全部程式碼 UTF-8，註解與 UI 文字用繁體中文；函式名英文。
- 不在 import 時載入模型；模型檔放 `models/` 子目錄（PyInstaller 時隨資料夾附帶），路徑用 `engine.paths.models_dir()` 解析（frozen 時用 `sys._MEIPASS` 或 exe 所在目錄）。
- 測試用 pytest，放 `tests/`，不得依賴網路；需要模型的測試用 `pytest.mark.slow` 標記並在無模型時 skip。
- 任何對外函式都要處理空輸入（沒有音符 → 產生只含休止符的一小節）。
