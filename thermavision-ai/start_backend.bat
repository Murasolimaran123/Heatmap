@echo off
echo Starting ThermaVision AI Backend (FastAPI)...
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
