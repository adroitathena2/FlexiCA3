<#
.SYNOPSIS
    Start the local Clef decision-model server (llama.cpp) for Paytriq.

.DESCRIPTION
    Serves the open-weight Cloudflare Clef-flash GGUF through llama.cpp's
    /v1/systemone endpoint. Fully local: no network, no rate limit, no cost,
    no API key. This is what makes the live demo reproducible offline.

    Two things this script exists to get right, both learned the hard way:

    1. MODEL. Use ggml-org/Clef-Flash-GGUF, NOT bartowski/Cloudflare_clef-flash-GGUF.
       The bartowski conversion is HEADLESS -- it contains only the Qwen3.5-9B
       backbone with Clef's decision head stripped out (verified: its own
       layouts/*.tensor-types.txt lists 250 tensors and zero `dec.*` head
       tensors, and it was quantized with llama.cpp b11279, before clef support
       landed). It serves prose on /v1/chat/completions and has no
       /v1/systemone route. Using it silently gives you a text model.

    2. ENDPOINT. Decisions are POSTed to /v1/systemone, never
       /v1/chat/completions. Both routes exist on this server; the wrong one
       returns text instead of probabilities.

    Requires llama.cpp >= b11364 (the `clef` architecture merge). Verified on
    build 11392.

.EXAMPLE
    .\scripts\serve_clef.ps1
    .\scripts\serve_clef.ps1 -Port 18782 -Context 8192

.EXAMPLE
    .\scripts\serve_clef.ps1 -Stop
#>
[CmdletBinding()]
param(
    [int]$Port = 18781,
    [int]$Context = 4096,
    [int]$Ubatch = 4096,
    [int]$GpuLayers = 99,
    [switch]$Stop,
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'llama\llama-server.exe'
$model = Join-Path $root 'models\Clef-Flash-Q4_K_M.gguf'
$logDir = Join-Path $root 'llama'
$url = "http://127.0.0.1:$Port"

function Stop-Clef {
    $procs = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -ErrorAction SilentlyContinue
    if (-not $procs) { Write-Host 'no llama-server.exe running'; return }
    foreach ($p in $procs) {
        Write-Host "stopping pid=$($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

if ($Stop) { Stop-Clef; exit 0 }

# ---------------------------------------------------------------- preflight
if (-not (Test-Path $exe)) {
    throw "llama-server.exe not found at $exe`n" +
          "Download the CUDA build (>= b11364) and extract BOTH zips to .\llama\`n" +
          "  llama-bXXXXXX-bin-win-cuda-12.4-x64.zip`n" +
          "  cudart-llama-bin-win-cuda-12.4-x64.zip"
}
if (-not (Test-Path $model)) {
    throw "model not found at $model`n" +
          "Download the HEAD-PRESERVING quant (Apache-2.0):`n" +
          "  https://huggingface.co/ggml-org/Clef-Flash-GGUF/resolve/main/Clef-Flash-Q4_K_M.gguf"
}

$versionLine = (& $exe --version 2>&1 | Select-Object -First 1 | Out-String).Trim()
Write-Host "llama.cpp build: $versionLine"

# Parse "build 11392" specifically. Stripping every non-digit from the whole line
# also swallows the commit SHA, producing a number far too large for Int32.
$b = 0
if ($versionLine -match 'build\s+(\d+)') { $b = [int]$Matches[1] }
if ($b -and $b -lt 11364) {
    Write-Warning "build $b predates the clef architecture merge (b11364); /v1/systemone will not exist. Upgrade llama.cpp."
}

$sizeGb = [math]::Round((Get-Item $model).Length / 1GB, 2)
Write-Host "model: $model ($sizeGb GiB)"
if ($sizeGb -lt 5.5) {
    Write-Warning "model is only $sizeGb GiB - this is likely the HEADLESS bartowski conversion, which cannot do System One decisions."
}

# ---------------------------------------------------------------- port check
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy -and -not $Restart) {
    $holder = $busy | ForEach-Object { (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName } | Select-Object -Unique
    Write-Warning "port $Port is already in use by: $($holder -join ', ')"
    Write-Warning "either stop it, or pick another port:  .\scripts\serve_clef.ps1 -Port 18782"
    Write-Warning "then set CLEF_BASE_URL=http://127.0.0.1:$Port in your .env"
    exit 1
}
if ($Restart) { Stop-Clef }

# ---------------------------------------------------------------- start
$out = Join-Path $logDir 'server.log'
$err = Join-Path $logDir 'server.err.log'
foreach ($f in @($out, $err)) { if (Test-Path $f) { Remove-Item $f -Force } }

Write-Host "starting clef on $url (ctx=$Context ubatch=$Ubatch ngl=$GpuLayers) ..."
$p = Start-Process -FilePath $exe -PassThru -WindowStyle Hidden `
    -ArgumentList '-m', $model, '-ngl', "$GpuLayers", '-c', "$Context", `
                  '-b', "$Ubatch", '-ub', "$Ubatch", '--host', '127.0.0.1', '--port', "$Port" `
    -RedirectStandardOutput $out -RedirectStandardError $err

$up = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 2
    try {
        if ((Invoke-WebRequest "$url/health" -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $up = $true; break }
    } catch { }
    if ($p.HasExited) {
        Write-Host "server exited with code $($p.ExitCode). Tail of stderr:" -ForegroundColor Red
        Get-Content $err -Tail 25 | ForEach-Object { "  $_" }
        exit 1
    }
}

if (-not $up) { Write-Host 'timed out waiting for /health' -ForegroundColor Red; Get-Content $err -Tail 25; exit 1 }

# Confirm it is genuinely a decision model, not a chat model in disguise.
$kind = (Get-Content $err | Select-String -Pattern 'decision model type:\s*(\S+)').Matches.Groups[1].Value
if ($kind) {
    Write-Host "decision model type: $kind" -ForegroundColor Green
    if ($kind -ne 'clef') { Write-Warning "expected clef, got '$kind'" }
} else {
    Write-Warning "could not confirm 'decision model type: clef' in the log - /v1/systemone may be unavailable"
}

Write-Host ''
Write-Host "READY  $url/v1/systemone" -ForegroundColor Green
Write-Host "set in .env:  CLEF_BASE_URL=$url"
Write-Host "logs: $out , $err"
Write-Host "stop:  .\scripts\serve_clef.ps1 -Stop"
Write-Host ''
Write-Host 'smoke test:'
Write-Host "  Invoke-RestMethod -Method Post -Uri '$url/v1/systemone' -ContentType 'application/json' -Body (@{"
Write-Host "    state   = 'Sponsor says the price is too high.'"
Write-Host "    questions = @{ objection = @{ type='noul'; instructions='Is the sponsor objecting to price?' } }"
Write-Host "  } | ConvertTo-Json -Depth 5)"
