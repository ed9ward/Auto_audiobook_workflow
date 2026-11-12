@echo off
cd /d "%~dp0"
set AUDIOBOOK_HIDE_CONSOLE=1
set PYVER=3.14
py -%PYVER% -c "" >nul 2>&1
if errorlevel 1 set PYVER=3.13
py -%PYVER% -c "" >nul 2>&1
if errorlevel 1 set PYVER=3.12
py -%PYVER% audiobook_master.py %*
if errorlevel 1 pause
