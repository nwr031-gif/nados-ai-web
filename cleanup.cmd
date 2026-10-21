@echo off
REM cleans scratch/test artifacts (never touches source, state or logs)
cd /d "%~dp0"
for %%F in (n[0-9]*.txt t[0-9]*.txt u[0-9]*.txt v[0-9]*.txt w[0-9]*.txt x[0-9]*.txt y[0-9]*.txt z[0-9]*.txt t[a-z]*.txt n[a-z]*.txt r[0-9]*.txt *-out.txt *_out.txt raw-*.txt *.json) do (
  if exist "%%F" del /q "%%F"
)
echo [cleanup] scratch artifacts removed
