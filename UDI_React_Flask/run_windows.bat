@echo off
setlocal
cd /d %~dp0

echo [UDI] Creating Python environment...
if not exist backend\.venv python -m venv backend\.venv
call backend\.venv\Scripts\activate
if errorlevel 1 goto :error

echo [UDI] Installing backend dependencies...
python -m pip install -r backend\requirements.txt
if errorlevel 1 goto :error

if not exist backend\.env (
  copy backend\.env.example backend\.env
)

echo [UDI] Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 goto :error

echo [UDI] Building React frontend...
call npm run build
if errorlevel 1 goto :error

cd ..\backend
echo [UDI] Starting Flask...
python app.py
goto :eof

:error
echo.
echo [UDI] Startup failed. The command above returned an error.
exit /b 1
