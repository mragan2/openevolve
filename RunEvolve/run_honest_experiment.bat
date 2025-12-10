@echo off
REM ========================================================
REM  MASSIVE GRAVITON - THE "GENESIS" RESET
REM ========================================================

REM 1. Set working directory
cd /d "C:\Users\Michal\Documents\GitHub\openevolve"

REM 2. KILL ZOMBIES
taskkill /F /IM python.exe >nul 2>&1

REM 3. DESTROY OLD ARTIFACTS
if exist "examples\solve\openevolve_output" (
    rmdir /s /q "examples\solve\openevolve_output"
)
if exist "examples\solve\__pycache__" (
    rmdir /s /q "examples\solve\__pycache__"
)
REM Cleanup previous runs of this script
if exist "examples\solve\genesis.py" (
    del "examples\solve\genesis.py"
)

REM 4. ACTIVATE VENV
call .venv\Scripts\activate

REM 5. RESET TO HONEST CODE (Using the cleaner)
echo [STATUS] Creating fresh scaffold...
python examples/solve/clean_start.py

REM 6. THE SILVER BULLET: RENAME THE FILE
REM This forces Python to treat it as a brand new module, ignoring all caches.
echo [STATUS] Renaming initial_program.py -> genesis.py to bypass cache...
ren "examples\solve\initial_program.py" "genesis.py"

REM 7. RUN EVOLUTION
echo.
echo [STATUS] STARTING EVOLUTION (Input: genesis.py)
echo [EXPECTATION] Score MUST be low (~0.26). Watch the logs!
echo.
python openevolve-run.py examples/solve/genesis.py examples/solve/evaluator.py --config examples/solve/config.yaml

echo.
echo Finished. Press any key to close.
pause >nul