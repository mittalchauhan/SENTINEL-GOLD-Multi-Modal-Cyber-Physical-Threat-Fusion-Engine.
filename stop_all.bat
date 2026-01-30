@echo off
echo Stopping all Python & Spark processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM java.exe >nul 2>&1
echo All Sentinel services stopped.
pause
