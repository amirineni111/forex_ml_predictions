# Fix Forex ML Prediction Scheduled Task
# Run this script as Administrator in PowerShell
# Right-click PowerShell → "Run as Administrator"
#
# Schedule: Mon-Fri at 6:15 PM (forex data arrives at 6 PM for 5 PM EST close)
# Fix: Adds StartWhenAvailable + battery-safe settings so missed runs are retried

Write-Host "Updating Forex ML Prediction Task Scheduler settings..." -ForegroundColor Yellow

$ProjectPath = "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"
$DailyScript = "$ProjectPath\scripts\run_daily_predictions.bat"

# Remove existing task
try {
    Unregister-ScheduledTask -TaskName "forex_ml_prediction" -Confirm:$false -ErrorAction Stop
    Write-Host "Removed existing task 'forex_ml_prediction'" -ForegroundColor Green
} catch {
    Write-Host "Note: Could not remove 'forex_ml_prediction': $_" -ForegroundColor DarkYellow
}

# Create task with Mon-Fri 6:15 PM schedule + reliability improvements
$DailyAction = New-ScheduledTaskAction -Execute $DailyScript -WorkingDirectory $ProjectPath
$DailyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "18:15"
$DailySettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$DailyPrincipal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest

Register-ScheduledTask `
    -TaskName "forex_ml_prediction" `
    -Action $DailyAction `
    -Trigger $DailyTrigger `
    -Settings $DailySettings `
    -Principal $DailyPrincipal `
    -Description "Run forex ML predictions Mon-Fri at 6:15 PM (after 6 PM data arrival, with venv activation)"

Write-Host "`nTask 'forex_ml_prediction' registered successfully!" -ForegroundColor Green

# Verify
Write-Host "`nVerification:" -ForegroundColor Yellow
Get-ScheduledTask -TaskName "forex_ml_prediction" | Format-Table TaskName, State -AutoSize
Get-ScheduledTask -TaskName "forex_ml_prediction" | Get-ScheduledTaskInfo | Format-List NextRunTime
Get-ScheduledTask -TaskName "forex_ml_prediction" | Select-Object -ExpandProperty Settings | Format-List StartWhenAvailable, AllowStartIfOnBatteries

Write-Host "Done! Task runs Mon-Fri at 6:15 PM with StartWhenAvailable enabled." -ForegroundColor Green
