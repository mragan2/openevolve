@echo off
REM Go to OpenEvolve repo root
cd /d "C:\Users\Michal\Documents\GitHub\openevolve"

REM Activate your virtual environment
call .venv\Scripts\activate

REM Set LLM endpoint for this shell session
set OPENAI_API_BASE=http://localhost:11434/v1

REM Load API key from environment or .env file
if not defined OPENAI_API_KEY (
    echo Warning: OPENAI_API_KEY is not set. Please set it before running experiments.
)

echo.
echo OpenEvolve environment is ready.
echo   OPENAI_API_BASE=%OPENAI_API_BASE%
echo   OPENAI_API_KEY is set.
echo.

REM Drop you into an interactive shell with env + venv active
cmd
