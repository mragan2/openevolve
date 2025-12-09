@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python openevolve-run.py examples/galaxy_rotation/initial_program.py examples/galaxy_rotation/evaluator.py --config examples/galaxy_rotation/config.yaml
pause
