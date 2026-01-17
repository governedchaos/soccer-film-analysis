@echo off
REM ============================================================
REM Soccer Film Analysis - Run Application (Windows)
REM ============================================================

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found.
    echo Run deploy.bat first to set up the project.
    pause
    exit /b 1
)

REM Check what to run
if "%~1"=="" goto gui
if "%~1"=="gui" goto gui
if "%~1"=="--gui" goto gui
if "%~1"=="-g" goto gui
if "%~1"=="cli" goto cli
if "%~1"=="--cli" goto cli
if "%~1"=="-c" goto cli
if "%~1"=="test" goto test
if "%~1"=="--test" goto test
if "%~1"=="-t" goto test
if "%~1"=="db" goto db
if "%~1"=="--db" goto db
if "%~1"=="-d" goto db
goto help

:gui
echo Starting GUI application...
python -m src.gui.main_window
goto end

:cli
shift
if "%~1"=="" (
    echo Usage: run.bat cli VIDEO_PATH [--depth quick^|standard^|deep]
    pause
    exit /b 1
)
echo Running CLI analysis...
python scripts\run_analysis.py %*
goto end

:test
echo Running tests...
python -c "from config.settings import settings; print(f'Database: {settings.db_name}'); print(f'Device: {settings.get_device()}'); print('Imports OK')"
pause
goto end

:db
echo Setting up database...
python scripts\setup_database.py
pause
goto end

:help
echo Soccer Film Analysis - Run Script
echo.
echo Usage: run.bat [command]
echo.
echo Commands:
echo   gui      Launch GUI application (default)
echo   cli      Run CLI analysis: run.bat cli VIDEO_PATH
echo   test     Run quick test
echo   db       Setup/reset database
echo.
pause
goto end

:end
