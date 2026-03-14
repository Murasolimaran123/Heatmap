@echo off
echo ================================================
echo   ThermaVision AI - Installation Script
echo ================================================

echo.
echo [1/3] Setting up Python backend...
cd /d "%~dp0backend"

python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo [2/3] Setting up Next.js frontend...
cd /d "%~dp0frontend"
call npm install

echo.
echo [3/3] Done!
echo.
echo To run the app:
echo   - Backend: run start_backend.bat
echo   - Frontend: run start_frontend.bat (in a separate terminal)
echo.
pause
