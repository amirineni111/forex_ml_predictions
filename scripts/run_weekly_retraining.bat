@echo off
REM Forex Weekly Model Retraining - Run every Sunday at 6:00 AM
cd /d "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"
python daily_forex_automation.py --retrain-now
echo Weekly retraining completed at %date% %time%