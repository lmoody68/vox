@echo off
title VOX Mesh Worker (listener only - no mic)
cd /d "%~dp0"
echo.
echo   VOX MESH WORKER
echo   ------------------------------------------------------------
echo   VOX runs here as a background team member with NO microphone,
echo   so it never fights the Voice Hub for the mic. JARVIS, Friday,
echo   and Claude can hand it tasks and it does them (browser, system
echo   scans, scheduled tasks, mail, calendar) and reports back.
echo.
echo   Say to JARVIS/Friday:  "ask Vox to run a system health check"
echo   Leave this window open.  Ctrl+C to stop.
echo.
python vox.py --serve
echo.
echo   VOX mesh worker stopped.  Press any key to close.
pause >nul
