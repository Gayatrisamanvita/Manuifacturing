@echo off
REM =============================================================================
REM FactorySmart AI - Launch Streamlit Interactive Executive Dashboard
REM =============================================================================

if exist "venv\Scripts\activate.bat" (
    echo [*] Activating local virtual environment...
    call venv\Scripts\activate.bat
)

streamlit run streamlit_app.py
pause
