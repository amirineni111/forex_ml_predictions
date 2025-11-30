@echo off
REM Forex Daily Predictions - Immediate execution (no delay)
REM Use this version if you schedule your predictions AFTER your data loading job

cd /d "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"
echo Starting forex predictions at %date% %time%
python daily_forex_automation.py --run-now
echo Daily predictions completed at %date% %time%