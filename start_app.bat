@echo off
REM Spectrum Analyzer Flask App Startup Script
REM This script starts the Flask-based spectrum analyzer web application

echo ========================================
echo   Spectrum Analyzer Flask Application
echo ========================================
echo.

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
    echo.
    echo TIP: Create a virtual environment with:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements_flask.txt
    echo.
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Starting Flask application...
echo.
echo The app will be available at:
echo   http://localhost:5000
echo   http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the Flask app
python flask_app.py

REM If the script exits, pause so user can see error messages
if errorlevel 1 (
    echo.
    echo ERROR: The application encountered an error
    pause
)
