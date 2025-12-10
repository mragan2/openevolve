@echo off
REM Set working directory
cd /d "C:\Users\Michal\Documents\GitHub\openevolve"

REM Activate virtual environment
call .venv\Scripts\activate

REM Run evolution
python openevolve-run.py examples/solve/6_d.py examples/solve/evaluator.py --config examples/solve/config.yaml

echo.
echo Finished. Press any key to close.
pause >nul
