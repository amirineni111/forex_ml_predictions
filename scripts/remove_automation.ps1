# Remove Forex Trading System scheduled tasks
# Run as Administrator in PowerShell

Write-Host "🗑️ Removing Forex Trading System automation tasks..." -ForegroundColor Red

try {
    # Remove the scheduled tasks
    Unregister-ScheduledTask -TaskName "Forex Daily Predictions" -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "✅ Removed: Forex Daily Predictions" -ForegroundColor Green
    
    Unregister-ScheduledTask -TaskName "Forex Weekly Retraining" -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "✅ Removed: Forex Weekly Retraining" -ForegroundColor Green
    
    Write-Host "`n🎯 All forex automation tasks have been removed." -ForegroundColor Green
} catch {
    Write-Host "❌ Error removing tasks: $($_.Exception.Message)" -ForegroundColor Red
}