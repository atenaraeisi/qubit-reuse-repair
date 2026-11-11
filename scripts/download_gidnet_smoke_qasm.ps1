# Run from project ROOT in PowerShell. These circuits and plans are in the
# original CaQR rows of reuse_audit_fixed/fixed_full/original_plans.csv.
$ErrorActionPreference = 'Stop'
$raw = 'https://raw.githubusercontent.com/atenaraeisi/GidNET-Qubit-Reuse-Algorithm/experiments'
New-Item -ItemType Directory -Force 'pilot_cases/GidNET' | Out-Null
foreach ($filename in @('graycode6_47.qasm', '4mod5-v0_18.qasm')) {
    $output = Join-Path 'pilot_cases/GidNET' $filename
    Write-Host "Fetching $filename ..."
    curl.exe -fL "$raw/benchmarks_compact/$filename" -o $output
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $filename" }
    if ((Get-Item $output).Length -lt 20) { throw "Unexpectedly short QASM: $filename" }
}
Write-Host 'GidNET smoke QASM (CaQR benchmark source) files ready.'
