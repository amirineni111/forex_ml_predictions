# Forex Trading System Automation Setup Guide

## ⏰ Important: Data Loading Timing

**Your system has another job that loads data into `forex_hist_data` table daily.** 

### Timing Options:

#### Option A: Fixed Schedule with Delay (Current Setup)
- **Data Loading Job**: Runs at your preferred time (e.g., 6:30 AM)
- **Forex Predictions**: Scheduled at 7:00 AM with 5-minute delay
- The prediction script waits 5 minutes to ensure data loading completes

#### Option B: Sequential Scheduling (Recommended)
- **Data Loading Job**: 6:30 AM  
- **Forex Predictions**: 7:05 AM (immediate execution)
- Use `run_daily_predictions_immediate.bat` instead

### Batch File Options:
- `scripts/run_daily_predictions.bat` - Includes 5-minute delay
- `scripts/run_daily_predictions_immediate.bat` - No delay (use with sequential scheduling)

## 🎯 Overview
This guide helps you set up automated scheduling for your forex ML trading system using Windows Task Scheduler.

## 📅 Automation Schedule
- **Daily Predictions**: Every day at 7:00 AM
- **Weekly Retraining**: Every Sunday at 6:00 AM

## 🚀 Quick Setup (Recommended)

### Option 1: Automatic Setup (PowerShell)
1. **Right-click** PowerShell and select **"Run as Administrator"**
2. Navigate to your project folder:
   ```powershell
   cd "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"
   ```
3. Run the automation setup script:
   ```powershell
   .\scripts\setup_automation.ps1
   ```

### Option 2: Manual Setup (Task Scheduler GUI)
1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Click **"Create Basic Task"** in the right panel
3. Follow the wizard for each task:

#### Daily Predictions Task:
- **Name**: Forex Daily Predictions
- **Trigger**: Daily at 7:00 AM
- **Action**: Start a program
- **Program**: `C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex\scripts\run_daily_predictions.bat`

#### Weekly Retraining Task:
- **Name**: Forex Weekly Retraining
- **Trigger**: Weekly on Sunday at 6:00 AM
- **Action**: Start a program
- **Program**: `C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex\scripts\run_weekly_retraining.bat`

## 📁 Files Created for Scheduling

### Batch Files (Windows Task Scheduler compatible)
- `scripts/run_daily_predictions.bat` - Daily predictions runner
- `scripts/run_weekly_retraining.bat` - Weekly retraining runner

### PowerShell Scripts (Automation helpers)
- `scripts/setup_automation.ps1` - Automatic task scheduler setup
- `scripts/remove_automation.ps1` - Remove scheduled tasks

## 🔍 Verify Setup
After setup, verify your tasks:
```powershell
Get-ScheduledTask -TaskName "Forex*" | Format-Table TaskName, State
```

## 📊 What Each Mode Does

### 1. Daily Predictions Mode (7:00 AM)
- Processes 10 currency pairs
- Generates 50 trading signals per pair
- Exports to SQL Server database tables:
  - `forex_ml_predictions` (individual signals)
  - `forex_daily_summary` (aggregated data)
- Creates CSV backup reports
- Takes ~3-5 minutes to complete

### 2. Weekly Retraining Mode (Sunday 6:00 AM)
- Downloads fresh forex data
- Retrains ML models with new data
- Updates model performance metrics
- Saves improved models to `./data/best_forex_model.joblib`
- Takes ~15-30 minutes to complete

## ⚙️ Manual Testing
Test your automation anytime:
```bash
# Test daily predictions
python daily_forex_automation.py --run-now

# Test weekly retraining
python daily_forex_automation.py --retrain-now
```

## 🛠️ Troubleshooting

### Task Not Running?
1. Check Task Scheduler: `taskschd.msc`
2. Verify task status and last run result
3. Check Windows Event Viewer for errors
4. Ensure Python is in your system PATH

### Database Connection Issues?
1. Verify SQL Server is running
2. Check connection string in `src/database/connection.py`
3. Test manual connection: `python predict_forex_signals.py --currency-pair EURUSD --days-back 5`

## 🔄 Updating Automation
To modify schedule times, edit the PowerShell script or use Task Scheduler GUI to modify existing tasks.

## 🗑️ Remove Automation
To remove all scheduled tasks:
```powershell
.\scripts\remove_automation.ps1
```