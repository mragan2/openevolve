@echo off
REM ========================================================
REM SETUP: Set Root Directory and Activate Venv
REM ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate

echo.
echo ========================================================
echo  CONTINUING EVOLUTION FROM BEST FOUND PROGRAM
echo ========================================================
echo.

REM Define paths for clarity
set "SEED_PROGRAM=examples\time_force_idea\openevolve_output\best\best_program.py"
set "EVALUATOR=examples\time_force_idea\evaluator.py"
set "CONFIG=examples\time_force_idea\config_2.yaml"

REM Check if the best program actually exists before running
if not exist "%SEED_PROGRAM%" (
    echo [ERROR] Could not find best_program.py at:
    echo %SEED_PROGRAM%
    echo.
    echo Please run the initial simulation first to generate the 'best' folder.
    pause
    exit /b
)

REM ========================================================
REM EXECUTE
REM ========================================================
echo Input Program: %SEED_PROGRAM%
echo Evaluator:     %EVALUATOR%
echo Config:        %CONFIG%
echo.

python openevolve-run.py "%SEED_PROGRAM%" "%EVALUATOR%" --config "%CONFIG%"

echo.
echo [DONE] Process finished.
pause