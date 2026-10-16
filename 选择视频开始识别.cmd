@echo off
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\run_video.py"
if errorlevel 1 (
  echo 处理未完成，请查看上方提示。
  pause
)
