@echo off
setlocal

REM ========================================================
REM CONFIGURATION
REM ========================================================
set "PROJECT_ROOT=%~dp0"
set "SEED_FILE=examples\time_force_idea\seed_for_phase1.py"
set "EVALUATOR=examples\time_force_idea\evaluator.py"
set "CONFIG=examples\time_force_idea\config_2.yaml"

REM ========================================================
REM SETUP
REM ========================================================
cd /d "%PROJECT_ROOT%"
call .venv\Scripts\activate

echo.
echo ========================================================
echo   STARTING NEW EVOLUTIONARY RUN
echo   Seed: %SEED_FILE%
echo ========================================================
echo.

REM Check if seed exists
if not exist "%SEED_FILE%" (
    echo [ERROR] Seed file not found at:
    echo %SEED_FILE%
    echo Please save your python code there first!
    pause
    exit /b
)

REM ========================================================
REM EXECUTE OPENEVOLVE
REM ========================================================
REM This takes the manual seed and starts the evolution process
python openevolve-run.py "%SEED_FILE%" "%EVALUATOR%" --config "%CONFIG%"

echo.
echo [DONE] Simulation batch finished.
pause