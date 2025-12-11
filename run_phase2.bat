@echo off
cd /d "%~dp0"
call .venv\Scripts\activate

python openevolve-run.py examples/time_force_idea/initial_program.py examples/time_force_idea/evaluator.py --config examples/time_force_idea/config_2.yaml --checkpoint "C:\Users\Michal\Documents\GitHub\openevolve\examples\time_force_idea\openevolve_output\checkpoints\checkpoint_300"

pause