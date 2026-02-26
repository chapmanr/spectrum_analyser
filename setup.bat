@echo off
REM First-time setup script for Spectrum Analyzer Flask App
REM This script creates a virtual environment and installs dependencies

echo ========================================
echo   Spectrum Analyzer - First Time Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if venv already exists
if exist venv (
    echo Virtual environment already exists.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q venv
    ) else (
        echo Skipping virtual environment creation.
        goto :install
    )
)

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

:install
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies from requirements_flask.txt...
echo This may take a few minutes...
echo.
python -m pip install --upgrade pip
pip install -r requirements_flask.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start the application, run:
echo   start_app.bat
echo.
echo Or manually:
echo   venv\Scripts\activate
echo   python flask_app.py
echo.
pause
