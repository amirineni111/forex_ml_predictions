@echo off
REM Forex Daily Predictions - Run at 7:00 AM daily
REM Wait a few minutes to ensure data loading job has completed
echo Waiting 1 minute for data loading to complete...
timeout /t 60 /nobreak

cd /d "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"
echo Starting forex predictions at %date% %time%
python daily_forex_automation.py --run-now
echo Daily predictions completed at %date% %time%