@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python openevolve-run.py examples/hubble_tension/initial_program.py examples/hubble_tension/evaluator.py --config examples/hubble_tension/config.yaml
pause
