@echo off
echo ===============================================
echo    Forex ML Automation with Signal Strength  
echo ===============================================
echo.
echo Select automation option:
echo 1. Run Daily Predictions (All Currency Pairs)
echo 2. Retrain Enhanced Model (Weekly Task)  
echo 3. Start Scheduled Automation Service
echo 4. Run Batch Predictions Now (All Pairs)
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting daily predictions with enhanced signal strength features...
    python daily_forex_automation.py --run-now
) else if "%choice%"=="2" (
    echo.
    echo Starting weekly model retraining with signal strength enhancement...
    python daily_forex_automation.py --retrain-now
) else if "%choice%"=="3" (
    echo.
    echo Starting scheduled automation service...
    echo Daily predictions: 7:00 AM
    echo Weekly retraining: Sunday 6:00 AM
    echo Press Ctrl+C to stop
    python daily_forex_automation.py
) else if "%choice%"=="4" (
    echo.
    echo Running batch predictions for all currency pairs...
    python run_all_forex_predictions.py --export-db
) else (
    echo Invalid choice. Exiting...
    goto end
)

:end
echo.
echo Press any key to exit...
pause >nul