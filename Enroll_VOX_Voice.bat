@echo off
title VOX - Enroll Your Voice (voiceprint)
cd /d "%~dp0"
echo.
echo   VOX VOICEPRINT ENROLLMENT
echo   ------------------------------------------------------------
echo   Teach VOX YOUR voice so hands-free mode ignores the TV and
echo   other people, and responds ONLY to you.
echo.
echo   You'll be asked to say 5 short phrases. Speak normally, the
echo   way you'll talk to VOX, in your usual spot near the mic.
echo   (First run downloads a small speaker model - one time.)
echo.
python vox.py --enroll
echo.
echo   Done. Launch "VOX - Hey Vox (hands-free)" and it will now
echo   only answer your voice.  Press any key to close.
pause >nul
