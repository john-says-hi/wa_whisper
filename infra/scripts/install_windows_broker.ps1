param([string]$AppDirectory = "$env:USERPROFILE\Documents\wa_whisper")
$ErrorActionPreference = "Stop"
$python = Join-Path $AppDirectory ".venv\Scripts\pythonw.exe"
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $python -Argument "-m wa_whisper.broker_server" -WorkingDirectory $AppDirectory
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "WA Whisper Model Broker" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "WA Whisper Model Broker"
Get-ScheduledTask -TaskName "WA Whisper Model Broker" | Select-Object TaskName,State
