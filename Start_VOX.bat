@echo off
title VOX Voice Agent
cd /d "%~dp0"
echo.
echo   Starting VOX -- the local voice agent (push-to-talk)...
echo   (loading Whisper takes a few seconds the first time)
echo.
echo   HOW TO USE:  Press ENTER, then speak. It answers when you pause.
echo                Type  q  then ENTER to quit.
echo.
python vox.py
echo.
echo   VOX stopped. Press any key to close.
pause >nul
