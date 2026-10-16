@echo off
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -X utf8 -m visualmap.web_app
if errorlevel 1 (
  echo 工作区启动失败，请查看上方提示。
  pause
)
