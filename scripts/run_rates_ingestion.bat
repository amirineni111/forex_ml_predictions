@echo off
REM Forex rate/yield ingestion from FRED -> forex_rates_daily
REM Run daily BEFORE predictions/retraining so rate-differential features are fresh.

cd /d "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Starting FRED rate ingestion at %date% %time%
REM Incremental refresh of the last ~10 days (handles late FRED revisions)
python scripts\seed_forex_rates.py --days-back 10
set EXITCODE=%errorlevel%
echo Rate ingestion completed at %date% %time% (exit code: %EXITCODE%)
exit /b %EXITCODE%
