@echo off
REM Set working directory
cd /d "C:\Users\Michal\Documents\GitHub\openevolve"

REM Activate virtual environment
call .venv\Scripts\activate

REM ---------------------------------------------------------
REM 1. Start Visualizer in a NEW separate window
REM ---------------------------------------------------------
echo Starting Visualizer in a separate window...
start "OpenEvolve Visualizer" cmd /k "call .venv\Scripts\activate & python scripts\visualizer.py"