@echo off
REM Nados 24/7 launcher — keeps the app + benchmark running
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
:loop
node supervisor.mjs >> "logs\supervisor.log" 2>&1
echo [%date% %time%] supervisor exited, restarting in 10s... >> "logs\supervisor.log"
timeout /t 10 /nobreak >nul
goto loop
