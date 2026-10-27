@echo off
REM =============================================================================
REM FactorySmart AI - Launch Manufacturing Intelligence API
REM =============================================================================

if exist "venv\Scripts\activate.bat" (
    echo [*] Activating local virtual environment...
    call venv\Scripts\activate.bat
)

python run_api.py --port 8000
pause
