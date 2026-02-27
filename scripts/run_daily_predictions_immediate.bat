@echo off
REM Forex Daily Predictions - Immediate execution (no delay)
REM Use this version if you schedule your predictions AFTER your data loading job

cd /d "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Starting forex predictions at %date% %time%
python daily_forex_automation.py --run-now
set EXITCODE=%errorlevel%
echo Daily predictions completed at %date% %time% (exit code: %EXITCODE%)
exit /b %EXITCODE%