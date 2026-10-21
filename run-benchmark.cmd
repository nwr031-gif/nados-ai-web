@echo off
REM Run the 100-question benchmark manually (resumable). Options: --once --limit=N --rescore
cd /d "%~dp0"
node benchmark\benchmark.mjs %*
pause
