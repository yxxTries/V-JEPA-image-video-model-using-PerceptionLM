@echo off
rem ===========================================================================
rem  run.bat - launcher for the JEPA bench. Always runs inside .\.venv
rem
rem  Double-click          -> menu
rem  run.bat extract clips\archery.mp4 --model L
rem  run.bat classify clips\bowling.mp4
rem  run.bat bench --sizes B L --frames 8 16
rem  run.bat llava_prep                -> 20k LLaVA-Pretrain images + cached V-JEPA features
rem  run.bat train_projector           -> train the V-JEPA to Qwen2.5 MLP projector
rem  run.bat caption photo.jpg         -> image/video to caption
rem  run.bat setup          -> runs setup.ps1
rem  run.bat shell          -> opens a cmd prompt with the venv active
rem ===========================================================================
setlocal EnableExtensions
title JEPA bench
cd /d "%~dp0"
set "ROOT=%~dp0"
set "VENV=%~dp0.venv"
set "PYTHONUTF8=1"

if /I "%~1"=="setup" goto :do_setup

rem ---- make sure the venv exists --------------------------------------------
if exist "%VENV%\Scripts\activate.bat" goto :activate
echo [run] No virtual environment at "%VENV%".
choice /C YN /M "[run] Run setup.ps1 now to create it"
if errorlevel 2 goto :no_venv
call :do_setup_inner
if not exist "%VENV%\Scripts\activate.bat" goto :no_venv

:activate
call "%VENV%\Scripts\activate.bat"
if errorlevel 1 goto :bad_venv

rem ---- verify that "python" really is the venv interpreter -------------------
set "PYPREFIX="
for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.prefix)" 2^>nul`) do set "PYPREFIX=%%P"
if not defined PYPREFIX goto :bad_venv
if /I not "%PYPREFIX%"=="%VENV%" (
    echo [run] WARNING: python resolves to "%PYPREFIX%", expected "%VENV%".
    echo [run] Using the venv interpreter directly.
    set "PATH=%VENV%\Scripts;%PATH%"
)
echo [run] venv active: %VENV%

if "%~1"=="" goto :menu
if /I "%~1"=="shell" goto :shell

rem ---- direct mode: run.bat <script> [args...] ------------------------------
set "SCRIPT=%~n1"
if not exist "%ROOT%%SCRIPT%.py" (
    echo [run] Unknown script "%~1". Choose: extract classify pca_viz similarity bench hooks llava_prep train_projector caption setup shell
    exit /b 2
)
set "ARGS="
:collect
shift
if "%~1"=="" goto :run_direct
set ARGS=%ARGS% "%~1"
goto :collect
:run_direct
python "%ROOT%%SCRIPT%.py"%ARGS%
set "RC=%errorlevel%"
endlocal & exit /b %RC%

rem ---- interactive menu ------------------------------------------------------
:menu
echo.
echo  ================= JEPA bench =================
echo   1  extract     image/video to embeddings
echo   2  classify    SSv2 top-5 actions
echo   3  pca_viz     PCA RGB maps per frame
echo   4  similarity  cosine similarity over a folder
echo   5  bench       VRAM and latency sweep
echo   6  hooks       per-layer activations via nnsight
echo   7  setup       re-run setup.ps1
echo   8  shell       cmd prompt with venv active
echo  -------- V-JEPA -^> Qwen2.5 captioner ---------
echo   D  llava_prep       download 20k LLaVA-Pretrain + cache V-JEPA features
echo   T  train_projector  train the MLP projector, encoder + LLM frozen
echo   C  caption          image/video to caption
echo   Q  quit
echo  ==============================================
choice /C 12345678DTCQ /N /M "Choose: "
set "SEL=%errorlevel%"
if "%SEL%"=="12" goto :end
if "%SEL%"=="0" goto :end
if "%SEL%"=="8" goto :shell
if "%SEL%"=="7" (call :do_setup_inner & goto :menu)
if "%SEL%"=="5" goto :menu_bench
if "%SEL%"=="9" (set "SCRIPT=llava_prep" & goto :menu_noinput)
if "%SEL%"=="10" (set "SCRIPT=train_projector" & goto :menu_noinput)
if "%SEL%"=="1" set "SCRIPT=extract"
if "%SEL%"=="2" set "SCRIPT=classify"
if "%SEL%"=="3" set "SCRIPT=pca_viz"
if "%SEL%"=="4" set "SCRIPT=similarity"
if "%SEL%"=="6" set "SCRIPT=hooks"
if "%SEL%"=="11" set "SCRIPT=caption"

set "IN="
if "%SCRIPT%"=="similarity" (
    set /p "IN=Folder of clips (drag it here): "
) else (
    set /p "IN=Video or image path (drag it here): "
)
if not defined IN goto :menu
set IN=%IN:"=%
if not exist "%IN%" (
    echo [run] Not found: "%IN%"
    goto :menu
)
set "EXTRA="
if "%SCRIPT%"=="caption" (
    set /p "EXTRA=Extra options, e.g. --video frames --max-new-tokens 40  [Enter = defaults]: "
) else (
    set /p "EXTRA=Extra options, e.g. --model B --frames 8  [Enter = defaults]: "
)
echo.
echo [run] python %SCRIPT%.py "%IN%" %EXTRA%
python "%ROOT%%SCRIPT%.py" "%IN%" %EXTRA%
echo.
echo [run] exit code %errorlevel%
pause
goto :menu

:menu_bench
set "EXTRA="
set /p "EXTRA=Bench options, e.g. --sizes B L g --frames 8 16 32 --random-weights  [Enter = defaults]: "
echo [run] python bench.py %EXTRA%
python "%ROOT%bench.py" %EXTRA%
echo.
echo [run] exit code %errorlevel%
pause
goto :menu

:menu_noinput
rem llava_prep / train_projector take no input path, only options
set "EXTRA="
if "%SCRIPT%"=="llava_prep" set /p "EXTRA=Options, e.g. --n 2000 --grid 24 --data C:\jepa_data  [Enter = 20k images, 12x12 grid]: "
if "%SCRIPT%"=="train_projector" set /p "EXTRA=Options, e.g. --micro-batch 2 --accum 16 --dtype bf16  [Enter = defaults]: "
echo [run] python %SCRIPT%.py %EXTRA%
python "%ROOT%%SCRIPT%.py" %EXTRA%
echo.
echo [run] exit code %errorlevel%
pause
goto :menu

:shell
echo [run] Venv shell. Type "exit" to leave.
cmd /k "prompt (jepa) $P$G"
goto :end

rem ---- setup -------------------------------------------------------------------
:do_setup
call :do_setup_inner
endlocal & exit /b %errorlevel%

:do_setup_inner
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%setup.ps1"
exit /b %errorlevel%

:no_venv
echo [run] Cannot continue without .venv. Run:  run.bat setup
pause
endlocal & exit /b 1

:bad_venv
echo [run] .venv exists but is broken (folder moved or Python removed).
echo [run] Fix:  run.bat setup   (or: powershell -ExecutionPolicy Bypass -File setup.ps1 -Recreate)
pause
endlocal & exit /b 1

:end
endlocal
exit /b 0
