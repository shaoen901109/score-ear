@echo off
REM 聽歌寫譜 - 開發用啟動腳本（使用 .venv 內的 Python）
chcp 65001 >nul
cd /d "%~dp0"
.venv\Scripts\python -m app.main
pause
