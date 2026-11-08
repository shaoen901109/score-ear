# score-ear（聽歌寫譜）

[English](README.md) | 繁體中文

> 📖 **圖文教學：** https://shao-en.com/blog/score-ear-guide.html ——如果你不知道該如何操作，請把這篇文章交給你的 Claude 或 ChatGPT。

上傳一段錄音，自動辨識音符，輸出五線譜與簡譜。在自己電腦上執行（本機網頁介面），錄音不會上傳到任何伺服器。支援獨奏，也支援多樂器編制（先用 AI 把人聲、鼓、貝斯、其他分開再各自轉譜）。

> 自動轉譜不會百分之百正確，產出的譜適合當作「打譜的起點」，再自己校正。

## 功能

- **輸入格式**：WAV / FLAC / OGG / MP3 / M4A / MP4
- **模式**
  - 獨奏：`piano`、`vocal`、`guitar`、`other`、`auto`
  - 多樂器：用 demucs 分成 vocals / drums / bass / other 四軌，各自轉譜
- **選項**：BPM 覆寫、拍號（自動或手動）、量化最小格、使用 GPU、抑制泛音誤判（預設開啟，只在獨奏 auto 與多樂器模式有作用）
- **輸出**：MusicXML（可匯入 MuseScore 等打譜軟體）、MIDI、簡譜 SVG、五線譜 SVG、JSON

## 需求

- Windows 11（作者只在 Windows 測過；程式本身沒有寫死 Windows，但安裝腳本是 PowerShell）
- Python 3.10 或 3.11（3.12 以上部分套件未驗證）
- 建議有 NVIDIA 顯示卡：鋼琴模式與多樂器模式在 GPU 上快很多；沒有 GPU 也能用 CPU 跑，只是比較慢
- 磁碟空間：虛擬環境加模型約數 GB

## 安裝

```powershell
git clone https://github.com/shaoen901109/score-ear.git
cd score-ear
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

`setup_env.ps1` 會建立 `.venv` 虛擬環境並安裝所有套件：

- 一般套件從官方 PyPI 安裝；網路慢時可指定鏡像：`setup_env.ps1 -Mirror <鏡像網址>`
- PyTorch 預設裝 CUDA 12.8 版（RTX 50 系列需要 cu128 以上）。**沒有 NVIDIA 顯示卡**的話，請先把腳本裡的 `$TORCH_INDEX` 改成 `https://download.pytorch.org/whl/cpu`
- 也可以參考 `requirements.txt` 手動安裝；`requirements-lock.txt` 是作者環境的鎖定版本

## 使用

```powershell
run_dev.bat
```

或 `.venv\Scripts\python -m app.main`。會啟動本機伺服器（預設 `http://localhost:8000`，被佔用時自動往後找可用埠）並開啟瀏覽器。參數：`--port <埠號>`、`--no-browser`。

在網頁上選擇音檔與模式，送出後等待辨識完成，就能預覽與下載各種格式。

### 首次執行會下載模型

| 模式 | 首次下載 | 存放位置 |
|---|---|---|
| 獨奏 `auto` / `vocal` / `guitar` / `other` | 不需下載（basic-pitch 模型隨套件安裝） | — |
| 獨奏 `piano` | 約 165 MB 鋼琴轉譜模型 | `%USERPROFILE%\piano_transcription_inference_data\` |
| 多樂器 | 約 350 MB demucs 分離模型 | torch hub 快取 |

模型由各套件自行下載，各有自己的授權條款（見下方「第三方元件」）。

## 技術路線

- 音源分離：demucs（4 軌）
- 多音轉譜：basic-pitch（ONNX 後端，不需要 TensorFlow）
- 鋼琴專用：piano_transcription_inference（ByteDance 的高解析度鋼琴轉譜模型）
- 單音精修：torchcrepe（人聲旋律音高追蹤）
- 節奏與速度：librosa onset 偵測，可手動覆寫 BPM
- 調性偵測：Krumhansl-Schmuckler 音級輪廓
- 五線譜渲染：verovio（MusicXML → SVG）
- 簡譜渲染：自製 SVG（級數、八度點、升降記號、附點、時值）

模組契約與資料模型見 `docs/ARCHITECTURE.md`；準確度評估見 `docs/ACCURACY.md`。

## 已知限制

- `auto` 模式對泛音敏感，容易多抓音；人聲建議用 `vocal`、鋼琴用 `piano`
- 鼓聲分類用頻帶能量規則（不是機器學習分類器）；在合成測試音檔上 F1 約 80%（每次執行會有 ±5–15 個百分點的波動），真實錄音中 hi-hat 與小鼓密集時仍可能誤判
- demucs 對人工合成音源分離不穩定；真實錄音表現較好
- 專案路徑含中文時，verovio 讀不到內建字型；程式已自動把字型資源複製到 `%LOCALAPPDATA%\ScoreEar\` 解決

## 測試

```powershell
.\.venv\Scripts\python -m pytest tests -q -m "not slow"
```

`slow` 標記的測試會下載模型，預設排除。

## 校驗下載檔

到 [Releases](https://github.com/shaoen901109/score-ear/releases) 下載 `score-ear-v1.0.0.zip` 和 `SHA256SUMS.txt`（版本號以你實際下載的為準）。

1. 用 PowerShell 比對指紋（SHA256，檔案的數位指紋）：

   ```powershell
   Get-FileHash .\score-ear-v1.0.0.zip -Algorithm SHA256
   ```

   輸出的 `Hash` 要和 `SHA256SUMS.txt` 裡的字串一致（大小寫不同沒關係）。

2. 驗證來源（需安裝 [GitHub CLI](https://cli.github.com/)）：

   ```powershell
   gh attestation verify .\score-ear-v1.0.0.zip --repo shaoen901109/score-ear
   ```

   顯示驗證成功，代表這個 zip 是 GitHub Actions 從本 repo 原始碼建出來的，中途沒有被替換。

GitHub 上標示 **Verified** 的 commit，是作者本人簽章。

## 技術來源與致謝

本專案建立在這些開源專案與研究之上：

| 用途 | 專案 | 授權 |
|---|---|---|
| 音源分離 | [Demucs](https://github.com/adefossez/demucs)（Meta AI Research） | MIT |
| 多音轉譜 | [Basic Pitch](https://github.com/spotify/basic-pitch)（Spotify） | Apache-2.0 |
| 鋼琴轉譜 | [piano_transcription_inference](https://github.com/qiuqiangkong/piano_transcription_inference)（Kong et al., ByteDance 高解析度鋼琴轉譜） | 見原專案 |
| 音高追蹤 | [torchcrepe](https://github.com/maxrmorrison/torchcrepe)（CREPE 的 PyTorch 實作） | MIT |
| 音訊分析 | [librosa](https://librosa.org/) | ISC |
| 五線譜渲染 | [Verovio](https://www.verovio.org/) | LGPL-3.0 |
| 深度學習框架 | [PyTorch](https://pytorch.org/)、[ONNX Runtime](https://onnxruntime.ai/) | BSD-3-Clause、MIT |
| 介面與伺服器 | [FastAPI](https://fastapi.tiangolo.com/)、[Uvicorn](https://www.uvicorn.org/)、[pywebview](https://pywebview.flowrl.com/) | MIT、BSD-3-Clause、BSD-3-Clause |
| MIDI | [mido](https://github.com/mido/mido)、[pretty_midi](https://github.com/craffel/pretty-midi) | MIT |
| 調性偵測演算法 | Krumhansl-Schmuckler 音級輪廓（Krumhansl, 1990） | — |

本 repo 只有自己的程式碼，不包含任何模型檔；模型權重由各套件從官方來源下載，適用各自的授權條款。請只轉譜你有權使用的錄音。

開發過程使用 [Claude Code](https://claude.com/claude-code)（Anthropic）協助。

## 作者

- 網站：https://shaoenstudio.com
- 聯絡信箱：shaoenstudio@gmail.com
- GitHub：[@shaoen901109](https://github.com/shaoen901109)

## 授權

MIT，見 [LICENSE](LICENSE)。
