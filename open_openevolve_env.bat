@echo off
REM Go to OpenEvolve repo root
cd /d "C:\Users\Michal\Documents\GitHub\openevolve"

REM Activate your virtual environment
call .venv\Scripts\activate

REM Set LLM endpoint and key for this shell session
set OPENAI_API_BASE=http://localhost:11434/v1
set OPENAI_API_KEY=aa249496fa974637a67ebe8f05be1e21.bfs5CdlZ_ocSK0O__Guty9w0

echo.
echo OpenEvolve environment is ready.
echo   OPENAI_API_BASE=%OPENAI_API_BASE%
echo   OPENAI_API_KEY is set.
echo.

REM Drop you into an interactive shell with env + venv active
cmd
