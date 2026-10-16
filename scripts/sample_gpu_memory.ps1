param([Parameter(Mandatory=$true)][int]$TargetProcessId,
      [Parameter(Mandatory=$true)][string]$OutputFile,
      [Parameter(Mandatory=$true)][string]$StopFile,
      [Parameter(Mandatory=$true)][string]$ReadyFile,
      [Parameter(Mandatory=$true)][string]$PhaseFile)
$ErrorActionPreference = 'Stop'
$taskClock = [Diagnostics.Stopwatch]::StartNew()
while (-not (Test-Path -LiteralPath $StopFile)) {
    $sampleStarted = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $phase = $null
    try { $phase = Get-Content -LiteralPath $PhaseFile -Raw | ConvertFrom-Json } catch {}
    try {
        $processRows = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
            Where-Object { $_.Name -like "pid_${TargetProcessId}_*" } |
            Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)
        $adapterRows = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
            Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)
        $device = (& nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>&1 | Out-String).Trim()
        $sample = [ordered]@{ sample_start_unix_ms=$sampleStarted; sample_end_unix_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();
            target_pid=$TargetProcessId; phase=$phase; process_memory=$processRows; adapter_memory=$adapterRows; nvidia_smi=$device; status='ok' }
    } catch {
        $sample = [ordered]@{sample_start_unix_ms=$sampleStarted;sample_end_unix_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();
            target_pid=$TargetProcessId;phase=$phase;status='error';error=$_.Exception.Message}
    }
    $sample | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath $OutputFile -Encoding utf8
    if (-not (Test-Path -LiteralPath $ReadyFile)) { 'ready' | Set-Content -LiteralPath $ReadyFile -Encoding utf8 }
    Start-Sleep -Milliseconds 200
}
