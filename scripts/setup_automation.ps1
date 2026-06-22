# Forex Trading System - Windows Task Scheduler Setup
# Run as Administrator in PowerShell

# Import Task Scheduler module
Import-Module ScheduledTasks

# Define paths
$ProjectPath = "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"
$RatesScript = "$ProjectPath\scripts\run_rates_ingestion.bat"
$DailyScript = "$ProjectPath\scripts\run_daily_predictions.bat"
$WeeklyScript = "$ProjectPath\scripts\run_weekly_retraining.bat"

Write-Host "🔧 Setting up Forex Trading System automation..." -ForegroundColor Green

# Create FRED Rate Ingestion Task (6:00 PM daily — after US bond close, before predictions)
# Refreshes forex_rates_daily so rate/yield-differential features are current.
$RatesAction = New-ScheduledTaskAction -Execute $RatesScript
$RatesTrigger = New-ScheduledTaskTrigger -Daily -At "18:00"
$RatesSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
$RatesPrincipal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName "Forex Rate Ingestion" -Action $RatesAction -Trigger $RatesTrigger -Settings $RatesSettings -Principal $RatesPrincipal -Description "Ingest FRED rate/yield data into forex_rates_daily daily at 6:00 PM" -Force

Write-Host "✅ Rate ingestion task scheduled for 6:00 PM" -ForegroundColor Green

# Create Daily Predictions Task (8:55 PM daily)
$DailyAction = New-ScheduledTaskAction -Execute $DailyScript
$DailyTrigger = New-ScheduledTaskTrigger -Daily -At "20:55"
$DailySettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$DailyPrincipal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName "Forex Daily Predictions" -Action $DailyAction -Trigger $DailyTrigger -Settings $DailySettings -Principal $DailyPrincipal -Description "Run forex ML predictions daily at 8:55 PM" -Force

Write-Host "✅ Daily predictions task scheduled for 8:55 PM" -ForegroundColor Green

# Create Weekly Retraining Task (Sunday 10:00 AM)
$WeeklyAction = New-ScheduledTaskAction -Execute $WeeklyScript
$WeeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "10:00"
$WeeklySettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 4)
$WeeklyPrincipal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName "Forex Weekly Retraining" -Action $WeeklyAction -Trigger $WeeklyTrigger -Settings $WeeklySettings -Principal $WeeklyPrincipal -Description "Retrain forex ML models weekly on Sunday at 10:00 AM" -Force

Write-Host "✅ Weekly retraining task scheduled for Sunday 10:00 AM" -ForegroundColor Green

# Display scheduled tasks
Write-Host "`n📋 Scheduled Tasks Created:" -ForegroundColor Yellow
Get-ScheduledTask -TaskName "Forex*" | Format-Table TaskName, State, @{Name="NextRunTime";Expression={$_.Triggers[0].StartBoundary}}

Write-Host "`n🎯 Setup Complete! Your forex system will now run automatically:" -ForegroundColor Green
Write-Host "  • Rate Ingestion:    Every day at 6:00 PM" -ForegroundColor Cyan
Write-Host "  • Daily Predictions: Every day at 8:55 PM" -ForegroundColor Cyan
Write-Host "  • Weekly Retraining: Every Sunday at 10:00 AM" -ForegroundColor Cyan
Write-Host "`n⚙️  To manage tasks: Open Task Scheduler (taskschd.msc)" -ForegroundColor Yellow