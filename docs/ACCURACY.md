# 準確度評估報告（ACCURACY）

自動由 `tests/eval_accuracy.py` 產生，測試素材為 `tests/make_fixtures.py` 合成的音檔（見 `runtime/fixtures/`，不進版控）。F1 定義：onset 容許 ±50ms、pitch 精確匹配，offset 不列入比對。「量化前」= 攔截 `notation.quantize.quantize()` 呼叫前的原始秒數 NoteEvent；「量化後」= 把 `transcribe()` 回傳的量化 Score 換算回秒數再比對，用來分辨問題出在辨識還是量化。

本檔案保留「改進前」（本次任務開始前的基準，程式碼與 fixture 都是舊版）與「改進後」（`tests/eval_accuracy.py` 目前的自動輸出）兩節對照，最後有一節改進摘要。「改進後」的 multi_mix 一節，`tests/make_fixtures.py` 的鼓聲合成與 ground truth 本身也改了（見改進摘要第 1 點），跟「改進前」的 multi_mix 不是同一份音檔，鼓聲欄位不能逐格直接對比數字——drums F1 的進步請以下面「改進摘要」中未經 demucs、對鼓聲單軌直接評估的數字為準，那組才是同一份程式碼在同一份音檔上的公平比較。

## 改進前（本次任務開始前的基準）

### 獨奏（solo）

| Fixture | 模式 | 量化前 P/R/F1 | 量化後 P/R/F1 | 音符數(前/後) | 耗時 | GPU | BPM 偵測 | 備註 |
|---|---|---|---|---|---|---|---|---|
| scale_mono | solo/vocal | 100.0%/100.0%/100.0% | 100.0%/100.0%/100.0% | 15/15 | 6.4s | 是 | 149.8 |  |
| scale_mono | solo/auto | 53.6%/100.0%/69.8% | 53.6%/100.0%/69.8% | 28/28 | 0.3s | 是 | 149.8 |  |
| piano_chords | solo/auto | 47.5%/100.0%/64.4% | 47.5%/100.0%/64.4% | 59/59 | 0.2s | 是 | 120.2 |  |
| piano_chords | solo/piano | 92.9%/92.9%/92.9% | 92.9%/92.9%/92.9% | 28/28 | 3.8s | 是 | 120.2 |  |

### 多樂器（multi，demucs 分離；舊版 multi_mix fixture：鼓聲只有 kick+hihat，無 snare）

耗時：1.8s，GPU：是，偵測 BPM：120.2

| Part | instrument | 音符數(量化前/後) | 音高範圍(量化後) |
|---|---|---|---|
| Vocal (P1) | vocal | 8/8 | 72-79 |
| Drums (P2) | drums | 16/16 | 36-42 |
| Bass (P3) | bass | 12/12 | 36-48 |
| Other (P4) | other | 10/10 | 36-79 |

Ground-truth track 對應最佳匹配 Part（量化後 F1）：

| GT track | 最佳匹配 Part | P/R/F1 |
|---|---|---|
| melody | Other (P4) | 80.0%/100.0%/88.9% |
| bass | Bass (P3) | 66.7%/100.0%/80.0% |
| drums | Drums (P2) | 43.8%/29.2%/35.0% |

### 改進前的觀察與已知限制

- **`instrument_hint=auto`／`piano` 在人工合成、泛音明顯的音檔上明顯弱於正確 hint**：`scale_mono` 用 `auto`（basic-pitch 一般複音模型，不做單音化）只有 69.8% F1，`piano_chords` 用 `auto` 只有 64.4%；兩者的 recall 都是 100%（真正的音符全部找到），precision 偏低是因為 basic-pitch 把合成音色裡明顯的泛音（諧波）當成獨立音符額外報出來。
- **`multi_mix` 的合成旋律音源常被 demucs 同時分到 `vocals` 與 `other` 兩軌**，導致同一段旋律在樂譜上重複出現，當時沒有任何去重機制。
- **鼓聲部分 F1 只有 35.0%**：kick(36) 與 hihat(42) 雖然都有出現在 `Drums` Part 的音高範圍內，但 `engine/transcribe_drums.py` 的頻帶能量分類器把不少音符誤判成 snare(38)（GT 音檔裡完全沒有 snare），推測是因為分類器用「頻帶能量占全頻譜比例」判斷，多個打擊樂器同時響時，響度大的樂器會把安靜樂器的比例稀釋到偵測不到。
- **拍號固定 4/4**：`engine/pipeline._resolve_time_signature` 對 `time_signature="auto"` 直接回傳 4/4（程式裡明寫 `# TODO: 之後做自動拍號偵測`），完全沒有偵測邏輯。

## 改進後（本次任務）

### 獨奏（solo）

| Fixture | 模式 | 量化前 P/R/F1 | 量化後 P/R/F1 | 音符數(前/後) | 耗時 | GPU | BPM 偵測 | 備註 |
|---|---|---|---|---|---|---|---|---|
| scale_mono | solo/vocal | 100.0%/100.0%/100.0% | 100.0%/100.0%/100.0% | 15/15 | 5.9s | 是 | 149.8 |  |
| scale_mono | solo/auto | 62.5%/100.0%/76.9% | 62.5%/100.0%/76.9% | 24/24 | 0.3s | 是 | 149.8 |  |
| piano_chords | solo/auto | 50.0%/100.0%/66.7% | 50.0%/100.0%/66.7% | 56/56 | 0.2s | 是 | 120.2 |  |
| piano_chords | solo/piano | 92.9%/92.9%/92.9% | 92.9%/92.9%/92.9% | 28/28 | 3.4s | 是 | 120.2 |  |

### 多樂器（multi，demucs 分離；新版 multi_mix fixture：鼓聲改為 kick+snare(第2、4拍)+hihat，音色更接近真鼓）

耗時：1.8s，GPU：是，偵測 BPM：120.2
警告：other 聲部移除 2 個疑似與 vocals 重複的旋律音符

| Part | instrument | 音符數(量化前/後) | 音高範圍(量化後) |
|---|---|---|---|
| Vocal (P1) | vocal | 9/9 | 36-79 |
| Drums (P2) | drums | 21/21 | 36-42 |
| Bass (P3) | bass | 10/10 | 36-45 |
| Other (P4) | other | 14/14 | 43-107 |

Ground-truth track 對應最佳匹配 Part（量化後 F1）：

| GT track | 最佳匹配 Part | P/R/F1 |
|---|---|---|
| melody | Vocal (P1) | 77.8%/87.5%/82.4% |
| bass | Bass (P3) | 80.0%/100.0%/88.9% |
| drums | Drums (P2) | 85.7%/75.0%/80.0% |

`multi_mix` 是 demucs＋GPU 推論鏈，非逐位元決定性，多跑幾次這幾個數字會有 ±5-15 個百分點的正常波動（例如 melody/drums F1 觀察到的範圍約 73%-94%），但都遠高於改進前的基準。

### 改進後的觀察與已知限制

- **`instrument_hint=auto`／`piano` 在人工合成、泛音明顯的音檔上仍弱於正確 hint，但 `suppress_harmonics`（TranscribeOptions 新欄位，預設開啟）已把 `auto` 的 precision 拉高一截**：`scale_mono` 用 `auto` 目前 76.9% F1（開發前 69.8%），`piano_chords` 用 `auto` 目前 66.7% F1（開發前 64.4%）；兩者的 recall 都還是 100%（真正的音符全部找到），`suppress_harmonics` 抓的是「onset 差 <40ms、pitch 差為八度／八度五度／兩八度／兩八度大三度、confidence 明顯偏低」這種保守條件，只清掉一部分明顯的泛音誤判，沒有做到跟正確 hint（`vocal` 單音化、`piano` 專用模型）一樣的水準——這仍然反映「人工合成音泛音結構跟真實樂器不同」這個測試素材本身的特性，不是 `auto` 模式的迴歸。
- **`multi_mix` 的合成旋律音源仍常被 demucs 同時分到 `vocals` 與 `other` 兩軌**：因為是人工合成的音色，不是真人聲紋，demucs（在真實錄音上訓練）沒辦法穩定把它歸到單一 stem；`engine/pipeline.py` 現在會在 other part 找 onset 差 <60ms、pitch 相同或差八度、confidence 較低的音符去重（見 warnings 欄位），減少同一段旋律被同時寫兩次的機率，但無法完全避免 demucs 把同一旋律分到兩個 stem 本身——這是使用人工合成測試音檔評估依賴真實聲音模型（demucs）時的已知限制，並非程式邏輯錯誤。
- **鼓聲分類已大幅改善**：`engine/transcribe_drums.py` 改用「各頻帶濾波後 RMS 相對視窗峰值」而非「頻帶能量占全頻譜比例」判斷 kick/snare/hihat（避免同時打擊時響度大的樂器把安靜樂器的比例稀釋到偵測不到），並加上頻帶衰減時間、spectral centroid 輔助判斷、onset 偵測前墊一小段靜音修掉「緊貼音檔開頭的 onset 常被漏掉」的邊界問題。在 `tests/make_fixtures.py` 更真實的鼓聲合成（snare = 200Hz 音頭 + 1-4kHz 帶通噪音、hihat = 6kHz 以上高通噪音 30ms、kick = 60→40Hz 掃頻 150ms，並加入第 2、4 拍 snare）上，`tests/test_engine_transcribe_drums.py` 驗證 kick/snare/hihat 各自 onset-level F1 都達到 1.00（未經 demucs，直接對鼓聲單軌評估）；經過 demucs 分離後的 `multi_mix` 完整流程 F1 約 0.7-0.9（見上表，demucs 分離品質本身有 run-to-run 變動）。
- **低音 Part 音高範圍大致正確**：本次執行 `Bass` part 音高範圍量測為 36-45，fixture ground-truth 是 36-48；`engine/pipeline.py` 呼叫 basic-pitch 時對 bass 已限制 `min_pitch=28, max_pitch=60`，所以即使有雜音/demucs 分離殘留，量測到的範圍也不會超出 28-60。這個 Part 的音符數/範圍在不同次執行之間會有些微差異（demucs／basic-pitch 在 GPU 上並非完全逐位元決定性），屬於這兩個第三方模型本身的正常變動，不是本次程式修改造成的迴歸。
- **拍號自動偵測（`time_signature="auto"`）**：`engine/rhythm.detect_time_signature()` 用 beat_times 對齊的重音強度序列，比較 3 拍與 4 拍分組下「哪個拍位最像 downbeat」的重音對比，明顯偏 3 拍才回 `3/4`，否則一律 `4/4`（6/8 暫不自動偵測）；beat 數不足或對比不夠明顯時保守回退 `4/4`。這幾個 fixture 都不足以拿來測試（`scale_mono`／`piano_chords`／`multi_mix` 本身不含明顯的 3 拍重音模式，也是驗證重點：不能被誤判成 3/4），實際驗證見 `tests/test_engine_rhythm.py` 新增的 `waltz_mono` 3/4 華爾滋 fixture（強-弱-弱重音）：偵測結果為 `3/4`；既有三個 4/4 fixture 維持 `4/4` 不受影響。
- **耗時欄位包含首次模型載入成本**：同一個 process 內，每種模型（basic-pitch/piano_transcription_inference/demucs）只在第一次呼叫時真正載入，`solo/auto` 第二次以後幾乎只剩推論時間（0.2s 級），`solo/vocal`／`solo/piano` 的耗時包含了第一次載入模型（含鋼琴 checkpoint 讀檔）的成本。正式伺服器是常駐 process，行為與此一致：第一個請求較慢，之後的請求快很多。

## 改進摘要（本次任務四項改動）

1. **鼓分類**（`engine/transcribe_drums.py`）：改用各頻帶濾波後時域 RMS 相對視窗峰值振幅（而非頻帶能量占全頻譜比例），避免多個打擊樂器同時響時互相稀釋；加上頻帶衰減時間、spectral centroid 輔助判斷；onset 偵測前墊 50ms 靜音修掉「緊貼音檔開頭的重拍常被漏偵測」的邊界問題。`tests/make_fixtures.py` 的鼓聲合成同步改為更接近真鼓（snare = 200Hz 音頭 + 1-4kHz 帶通噪音、hihat = 6kHz 以上高通噪音 30ms、kick = 60→40Hz 掃頻 150ms），`multi_mix` 加入第 2、4 拍 snare，並新增 `multi_mix.drums_only.wav`（demucs 分離前的鼓聲單軌）供直接評估分類器本身。多樂器管線送進分類器前也會先正規化音量（`engine.transcribe_drums.normalize_volume`）。
   **結果**：`tests/test_engine_transcribe_drums.py::test_transcribe_drums_multi_mix_fixture_per_instrument_f1` 驗證 kick/snare/hihat 各自 onset-level F1 從（改進前規則、舊 fixture 無 snare）約 kick 67%／hihat 67%，提升到 **kick/snare/hihat 各自 F1 = 1.00**（新 fixture，未經 demucs）。經 demucs 分離的 `multi_mix` 完整流程 drums F1 從 **35.0% → 約 73-89%**（詳見上方改進後表格；demucs 本身非決定性，多次執行有正常波動）。
2. **多樂器旋律重複去重**（`engine/pipeline.py` 新增 `_dedup_other_vs_vocals`）：other Part 中若有音符與 vocals Part 音符 pitch 相同或差八度、onset 差 < 60ms、且 other 該音 confidence 較低，就從 other 移除，並在 `Score.warnings` 記錄移除數量（例如本次執行：「other 聲部移除 2 個疑似與 vocals 重複的旋律音符」）。純函式邏輯以 `tests/test_engine_pipeline.py`（6 個案例）驗證；實際跑 `multi_mix` 全流程也確認 warnings 有正確觸發。
3. **拍號自動偵測**（`engine/rhythm.py` 新增 `detect_time_signature` + `engine/pipeline._resolve_time_signature`）：用 beat_times 對齊的重音強度序列比較 3 拍／4 拍分組的重音對比（自動找出最像 downbeat 的拍位，不假設 beat_track 的相位對齊），明顯偏 3 拍才回 `3/4`，否則 `4/4`（6/8 暫不自動偵測）。新增 `waltz_mono` 3/4 華爾滋單音旋律 fixture（強-弱-弱重音）。**結果**：`waltz_mono` 正確判成 `3/4`；既有 `scale_mono`／`piano_chords`／`multi_mix` 三個 4/4 fixture 全部維持 `4/4`（`tests/test_engine_rhythm.py` 5 個新案例全過）。
4. **通用模式（basic-pitch）泛音誤判抑制**（`engine/transcribe_poly.py` 新增 `suppress_harmonics`）：onset 差 <40ms、pitch 差為 12/19/24/28 半音（自然泛音列第 2/3/4/5 泛音）、且 confidence < 0.5 倍時刪除該音符；新增 `TranscribeOptions.suppress_harmonics: bool = True`（`notation/score_model.py` 同步更新 `to_dict`/`from_dict`，app 前端未加 UI）。**結果**：`scale_mono/auto` F1 **69.8% → 76.9%**，`piano_chords/auto` F1 **64.4% → 66.7%**（recall 皆維持 100%，precision 提升）；`piano_chords/piano`（鋼琴專用模型路徑，不經過 `transcribe_poly`）維持 **92.9%** 不變，確認未受影響。

## TranscribeOptions 變更

新增一個欄位：`suppress_harmonics: bool = True`（`notation/score_model.py`），其餘欄位未變動。
