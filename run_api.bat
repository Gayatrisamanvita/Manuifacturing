@echo off
REM =============================================================================
REM FactorySmart AI - Launch Streamlit Dashboard & Intelligence API
REM =============================================================================

if exist "venv\Scripts\activate.bat" (
    echo [*] Activating local virtual environment...
    call venv\Scripts\activate.bat
)

python run_api.py --api-port 8001
pause
