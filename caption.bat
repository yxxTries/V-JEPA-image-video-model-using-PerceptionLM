@echo off
rem caption.bat - image or video -> caption (V-JEPA 2.1 -> trained projector -> Qwen2.5-1.5B-Instruct)
rem   Double-click, then drag images/videos into the window (Enter on an empty line quits).
rem   Or drop files onto caption.bat, or:  caption.bat photo.jpg clip.mp4
rem   Extra options (e.g. --video frames):  run.bat caption clip.mp4 --video frames
setlocal
cd /d "%~dp0"
title JEPA caption
set "PYTHONUTF8=1"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [caption] No .venv here. Run: run.bat setup
    pause
    exit /b 1
)

rem ---- files given: caption each one, then wait so the window stays open
if "%~1"=="" goto :ask
:next
"%PY%" caption.py "%~1"
echo.
shift
if not "%~1"=="" goto :next
pause
exit /b

rem ---- no files: keep asking (each run loads the models, ~20 s)
:ask
set "IN="
set /p "IN=Image or video (drag it here, Enter to quit): "
if not defined IN exit /b
set "IN=%IN:"=%"
"%PY%" caption.py "%IN%"
echo.
goto :ask
