[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    python -m unittest -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }

    python -m PyInstaller --noconfirm --clean --onefile --windowed --name VAGPR-Return vagpr.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }

    $exe = (Resolve-Path -LiteralPath 'dist/VAGPR-Return.exe').Path
    $smoke = Join-Path (Resolve-Path -LiteralPath 'build').Path 'smoke-result.json'
    $process = Start-Process -FilePath $exe -ArgumentList "--self-test --report `"$smoke`"" `
        -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "Packaged smoke test failed ($($process.ExitCode))." }
    $report = Get-Content -LiteralPath $smoke -Raw | ConvertFrom-Json
    if ($report.source -ne 'synthetic self-test' -or $report.forecast_method -ne 'vagpr-return-v2') {
        throw 'Packaged smoke test returned unexpected output.'
    }
    Get-FileHash -LiteralPath $exe -Algorithm SHA256
}
finally {
    Pop-Location
}
