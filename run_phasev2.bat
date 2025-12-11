@echo off
cd /d "%~dp0"
call .venv\Scripts\activate

python openevolve-run.py examples/time_force_idea/initial_program_1.py examples/time_force_idea/eval.py --config examples/time_force_idea/config.yaml

pause
