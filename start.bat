@echo off
echo ============================================
echo   NapsGen - Custom DDPM Image Generator
echo ============================================
echo.

echo Starting FastAPI backend on port 8000...
start "NapsGen Backend" cmd /k "cd /d %~dp0 && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting Next.js frontend on port 3000...
start "NapsGen Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo Both servers starting...
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo   API docs: http://localhost:8000/docs
echo.
echo Press any key to exit this window (servers keep running).
pause > nul
