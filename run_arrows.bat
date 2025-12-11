@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python openevolve-run.py examples/arrows_of_time/initial_program.py examples/arrows_of_time/evaluator.py --config examples/arrows_of_time/config.yaml
pause
