@echo off
setlocal enabledelayedexpansion
:: ===========================================================================
:: Model Breeder — launcher (Windows)
::
:: Mirrors the launcher pattern from AUTOMATIC1111/Forge (webui.bat) and
:: ComfyUI's run scripts: creates a virtual environment on first run,
:: installs dependencies only when requirements.txt has actually changed,
:: creates the input/output workspace folders, then hands off to app.py
:: with whatever flags you set in webui-user.bat.
::
:: Customize launch behavior in webui-user.bat, not this file.
:: ===========================================================================
cd /d "%~dp0"

set "FORCE_REINSTALL=0"
set "SKIP_VENV_FLAG=0"
set "EXTRA_ARGS="
for %%a in (%*) do (
    if /I "%%a"=="--reinstall" (
        set "FORCE_REINSTALL=1"
    ) else if /I "%%a"=="--skip-venv" (
        set "SKIP_VENV_FLAG=1"
    ) else (
        set "EXTRA_ARGS=!EXTRA_ARGS! %%a"
    )
)

echo ############################################################
echo   Model Breeder - Gradio Edition
echo ############################################################

if exist webui-user.bat call webui-user.bat

if not defined PYTHON set "PYTHON=python"
if not defined USE_VENV set "USE_VENV=1"
if "%SKIP_VENV_FLAG%"=="1" set "USE_VENV=0"
if not defined VENV_DIR set "VENV_DIR=venv"

where %PYTHON% >nul 2>nul
if errorlevel 1 (
    echo ERROR: '%PYTHON%' not found on PATH.
    echo        Install Python 3.10+ from python.org and make sure
    echo        "Add python.exe to PATH" is checked during setup.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('%PYTHON% --version 2^>^&1') do echo Python:  %%v
echo Workdir: %cd%

:: ---- sanity: this looks like a complete checkout -----------------------
:: Checked here, before any venv creation or pip work: a missing
:: requirements.txt would otherwise only surface as a confusing pip error
:: after already spending time creating a virtual environment first.
set "MISSING="
if not exist "requirements.txt" set "MISSING=!MISSING! requirements.txt"
if not exist "app.py" set "MISSING=!MISSING! app.py"
if not exist "setup_workspace.py" set "MISSING=!MISSING! setup_workspace.py"
if not "%MISSING%"=="" (
    echo ERROR: this doesn't look like a complete Model Breeder checkout -
    echo        missing:%MISSING%
    echo        Make sure you copied/cloned the whole project, not just
    echo        individual files, and that you're running this script
    echo        from inside that folder.
    pause
    exit /b 1
)

if "%USE_VENV%"=="1" (
    if not exist "%VENV_DIR%\Scripts\python.exe" (
        echo Creating virtual environment in .\%VENV_DIR% ...
        %PYTHON% -m venv "%VENV_DIR%"
        if errorlevel 1 (
            echo ERROR: failed to create a virtual environment.
            pause
            exit /b 1
        )
    )
    call "%VENV_DIR%\Scripts\activate.bat"
    set "PYTHON=python"
    echo Virtual environment active: %VENV_DIR%
) else (
    echo Skipping virtual environment ^(USE_VENV=0 or --skip-venv^)
)

:: ---- dependencies: only reinstall when requirements.txt changed ----------
if "%USE_VENV%"=="1" (
    set "STATE_DIR=%VENV_DIR%"
) else (
    set "STATE_DIR=."
)
set "REQ_HASH_FILE=%STATE_DIR%\.requirements_hash"

set "NEW_HASH="
for /f "skip=1 tokens=* delims=" %%h in ('certutil -hashfile requirements.txt SHA256 2^>nul') do (
    if not defined NEW_HASH (
        echo %%h | findstr /r "^[0-9a-fA-F]*$" >nul && set "NEW_HASH=%%h"
    )
)

set "OLD_HASH="
if exist "%REQ_HASH_FILE%" set /p OLD_HASH=<"%REQ_HASH_FILE%"

if not "%NEW_HASH%"=="%OLD_HASH%" (set "NEEDS_INSTALL=1") else (set "NEEDS_INSTALL=0")
if "%NEW_HASH%"=="" set "NEEDS_INSTALL=1"
if "%FORCE_REINSTALL%"=="1" set "NEEDS_INSTALL=1"

if "%NEEDS_INSTALL%"=="1" (
    echo Installing/updating dependencies ^(this can take a while the first time^)...
    %PYTHON% -m pip install --upgrade pip --quiet
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: dependency installation failed - see the pip output above.
        pause
        exit /b 1
    )
    if not "%NEW_HASH%"=="" (
        if not exist "%STATE_DIR%" mkdir "%STATE_DIR%" >nul 2>nul
        echo %NEW_HASH%> "%REQ_HASH_FILE%"
    )
) else (
    echo Dependencies already up to date - skipping install.
    echo ^(run webui.bat --reinstall to force a reinstall^)
)

%PYTHON% setup_workspace.py

echo ############################################################
echo   Launching - close this window, or Ctrl+C, to stop
echo ############################################################
%PYTHON% app.py %COMMANDLINE_ARGS% %EXTRA_ARGS%

pause
