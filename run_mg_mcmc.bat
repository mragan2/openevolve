@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python examples\mg_mcmc\code\mg_mcmc_emcee.py
pause
