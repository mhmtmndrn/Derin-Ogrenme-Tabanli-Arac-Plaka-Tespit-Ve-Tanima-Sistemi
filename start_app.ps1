$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$candidates = @(
    @{ Command = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"; Args = @("app.py"); VersionArgs = @("--version") },
    @{ Command = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"; Args = @("app.py"); VersionArgs = @("--version") },
    @{ Command = "python"; Args = @("app.py"); VersionArgs = @("--version") },
    @{ Command = "py"; Args = @("-3", "app.py"); VersionArgs = @("-3", "--version") }
)

foreach ($candidate in $candidates) {
    $command = $candidate.Command
    $exists = Test-Path $command
    $onPath = $false
    if (-not $exists) {
        $onPath = [bool](Get-Command $command -ErrorAction SilentlyContinue)
    }

    if ($exists -or $onPath) {
        & $command @($candidate.VersionArgs) *> $null
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        & $command @($candidate.Args) @args
        exit $LASTEXITCODE
    }
}

Write-Host "Python bulunamadi." -ForegroundColor Red
Write-Host "Once Python kurup su komutlari calistirin:"
Write-Host "python -m venv .venv"
Write-Host ".\.venv\Scripts\activate"
Write-Host "pip install -r requirements.txt"
exit 1
