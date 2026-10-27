@echo off
REM =============================================================================
REM FactorySmart AI - Automated Virtual Environment Setup for Windows
REM =============================================================================

echo ================================================================================
echo   Setting up Virtual Environment (venv) for Manufacturing ML Suite...
echo ================================================================================

REM Check for python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH! Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [*] Python detected:
python --version

REM Create venv if not already existing
if not exist "venv" (
    echo [*] Creating virtual environment in .\venv ...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [*] Virtual environment 'venv' already exists.
)

REM Activate and install
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo [*] Installing requirements from requirements.txt...
pip install -r requirements.txt

echo ================================================================================
echo   Setup Complete!
echo   To activate your environment in cmd:
echo       call venv\Scripts\activate
echo   To launch the API:
echo       python run_api.py
echo   To launch the Streamlit Dashboard:
echo       streamlit run streamlit_app.py
echo ================================================================================
pause
