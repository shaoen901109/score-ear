# setup_env.ps1 - 自動建立「聽歌寫譜」開發環境 (Windows / PowerShell)
# 用法: 在本資料夾下執行
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
#
# pip 套件來源預設官方 PyPI；網路慢時可用 -Mirror 參數改成鏡像。

# PowerShell 變數不分大小寫，下方的 $MIRROR 就是這個參數
param([string] $Mirror = "https://pypi.org/simple")

$ErrorActionPreference = "Continue"
# RTX 50 系列 (Blackwell, sm_120) 需要 cu128 以上版本才有對應 kernel；
# 純 CPU 機器請把下面這行改成 https://download.pytorch.org/whl/cpu
$TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

Write-Host "=== 1. 選擇 Python (優先 3.10 / 3.11) ==="
$pyLauncher = $null
foreach ($ver in @("3.10", "3.11")) {
    & py "-$ver" -c "print(1)" *> $null
    if ($LASTEXITCODE -eq 0) { $pyLauncher = "-$ver"; break }
}

if (-not (Test-Path ".venv")) {
    if ($pyLauncher) {
        Write-Host "使用 py $pyLauncher"
        & py $pyLauncher -m venv .venv
    } else {
        Write-Host "找不到 Python 3.10/3.11，改用預設 python（版本可能是 3.12+，部分套件相容性未在本機驗證過，遇到問題請回報）"
        & python -m venv .venv
    }
} else {
    Write-Host ".venv 已存在，略過建立"
}

$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip

Write-Host "=== 2. 基礎音訊/MIDI/Web/桌面/打包/測試 套件 ==="
& $PY -m pip install -i $MIRROR `
    numpy scipy soundfile mido pretty_midi librosa imageio-ffmpeg `
    fastapi uvicorn python-multipart httpx pywebview pyinstaller pytest verovio

Write-Host "=== 3. PyTorch (CUDA 12.8 wheel index) ==="
& $PY -m pip install torch torchaudio --index-url $TORCH_INDEX

Write-Host "=== 4. 多樂器辨識 / 鋼琴轉譜 / 音源分離 / 音高追蹤 ==="
& $PY -m pip install -i $MIRROR "basic-pitch[onnx]"
& $PY -m pip install -i $MIRROR piano_transcription_inference demucs
& $PY -m pip install -i $MIRROR torchcrepe

Write-Host "=== 5. 驗證 torch 是否仍是 CUDA 版（demucs/piano_transcription_inference 的相依可能覆蓋成 CPU 版）==="
& $PY -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
$torchCheck = & $PY -c "import torch; print(torch.cuda.is_available())"
if ($torchCheck -notmatch "True" -and $pyLauncher) {
    Write-Host "偵測到 torch 被降級為 CPU 版，重新安裝 CUDA 版..."
    & $PY -m pip install torch torchaudio --index-url $TORCH_INDEX --force-reinstall
}

Write-Host "=== 完成 ==="
Write-Host "鎖定版本可用: .\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt"
