@echo off
REM Forex Weekly Model Retraining - Run every Sunday at 10:00 AM
cd /d "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"

REM Activate virtual environment
call .venv\Scripts\activate.bat

python daily_forex_automation.py --retrain-now
set EXITCODE=%errorlevel%
echo Weekly retraining completed at %date% %time% (exit code: %EXITCODE%)
exit /b %EXITCODE%