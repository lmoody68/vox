@echo off
title VOX Voice Agent (Hey Vox - hands-free)
cd /d "%~dp0"
echo.
echo   Starting VOX -- HANDS-FREE wake-word mode...
echo   (loading Whisper takes a few seconds the first time)
echo.
echo   HOW TO USE:  Just say  "Hey Vox"  then your question -- no button to press.
echo                Example:  "Hey Vox, what time is it?"
echo                Say  "Hey Vox, goodbye"  or press Ctrl+C to stop.
echo.
python vox.py --wake
echo.
echo   VOX stopped. Press any key to close.
pause >nul
