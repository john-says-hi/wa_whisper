param([string]$AppDirectory = "$env:USERPROFILE\Documents\wa_whisper")
$ErrorActionPreference = "Stop"
$python = Join-Path $AppDirectory ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $python)) { throw "Install the Windows virtual environment first." }
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $python -Argument "-m wa_whisper.windows_controller" -WorkingDirectory $AppDirectory
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "WA Whisper Voice Typing" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "WA Whisper.lnk"))
$shortcut.TargetPath = $python
$shortcut.Arguments = "-m wa_whisper.windows_controller"
$shortcut.WorkingDirectory = $AppDirectory
$shortcut.Description = "Voice to text. Ctrl+Shift+F1 toggles power; hold Right Alt to dictate."
$shortcut.Save()
Start-ScheduledTask -TaskName "WA Whisper Voice Typing"
Get-ScheduledTask -TaskName "WA Whisper Voice Typing" | Select-Object TaskName,State
