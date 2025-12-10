@echo off
REM ========================================================
REM  MASSIVE GRAVITON LAUNCHER (NUCLEAR RESET + RUN)
REM ========================================================

REM 1. Set working directory
cd /d "C:\Users\Michal\Documents\GitHub\openevolve"

REM 2. Activate virtual environment
call .venv\Scripts\activate

REM 3. Execute the Python Reset Script
REM This kills zombies, deletes the DB, and writes the correct code.
python reset_now.py

REM 4. Run OpenEvolve
echo.
echo [STATUS] Starting OpenEvolve...
echo [TARGET] Look for rho_q_today_score jumping up from 0.06!
echo.
python openevolve-run.py examples/mtdc_hubble/initial_program.py examples/mtdc_hubble/evaluator.py --config examples/mtdc_hubble/config.yaml

echo.
echo Finished. Press any key to close.
pause >nul