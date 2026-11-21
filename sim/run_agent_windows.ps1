# Run the operator agent with the Windows-installed (logged-in) Claude Code; the simulator MCP
# server runs inside WSL over wsl.exe stdio. Opens the live dashboard, then the replay at the end.
#   powershell -ExecutionPolicy Bypass -File sim\run_agent_windows.ps1 [-SimDistro Ubuntu] [-SimRecordDir D:\x]
#       [-SimServerEnv "ROBOT_SIM_PYTHON=/path/python NORMA_CORE_REPO=/path"] [claude args...]
# Claude gets ONLY the elrobot-sim tools: no shell/file tools, no other MCP servers, empty working dir.
# Arguments are parsed by hand (no param block): PowerShell would otherwise prefix-match Claude
# flags such as -p against script/common parameters. -Sim* options are ours, the rest go to claude.
$SimDistro = 'Ubuntu'; $SimRecordDir = ''; $SimServerEnv = ''; $SimRunSh = ''; $SimPort = 8891; $SimNoBrowser = $false
$ClaudeArgs = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    switch -CaseSensitive ($args[$i]) {
        '-SimDistro'    { $SimDistro = $args[++$i] }
        '-SimRecordDir' { $SimRecordDir = $args[++$i] }
        '-SimServerEnv' { $SimServerEnv = $args[++$i] }   # extra KEY=VALUE pairs for the server, space-separated
        '-SimRunSh'     { $SimRunSh = $args[++$i] }       # WSL path of a run.sh that knows 'sim-arm-mcp'
        '-SimPort'      { $SimPort = [int]$args[++$i] }
        '-SimNoBrowser' { $SimNoBrowser = $true }
        default         { $ClaudeArgs += [string]$args[$i] }
    }
}
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
if (-not $SimRecordDir) { $SimRecordDir = Join-Path $here 'sessions' }
New-Item -ItemType Directory -Force -Path $SimRecordDir | Out-Null
function To-Wsl([string]$p) { (wsl.exe -d $SimDistro -- wslpath -a ($p -replace '\\', '/')).Trim() }
$Server = if ($SimRunSh) { @('bash', $SimRunSh, 'sim-arm-mcp') } else { @('bash', "$(To-Wsl $here)/run_server.sh") }

$wslArgs = @('-d', $SimDistro, '--', 'env', 'ROBOT_SIM_GUI=1', "ROBOT_SIM_RECORD_DIR=$(To-Wsl $SimRecordDir)",
          "ROBOT_SIM_DASHBOARD_PORT=$SimPort") + @($SimServerEnv -split ' ' | Where-Object { $_ }) + $Server
$config = Join-Path $env:TEMP "elrobot-sim-mcp-$PID.json"
@{ mcpServers = @{ 'elrobot-sim' = @{ command = 'wsl.exe'; args = $wslArgs } } } | ConvertTo-Json -Depth 5 |
    Set-Content -Encoding ascii $config
$work = Join-Path $env:TEMP "elrobot-sim-agent-$PID"
New-Item -ItemType Directory -Force -Path $work | Out-Null

if (-not $SimNoBrowser) {
    # Open the dashboard of THIS run's session (another sim server may already hold the default port).
    Start-Job -ArgumentList $SimRecordDir, (Get-Date) -ScriptBlock {
        param($dir, $since)
        for ($i = 0; $i -lt 240; $i++) {
            $s = Get-ChildItem -Path $dir -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.CreationTime -ge $since.AddSeconds(-2) } | Sort-Object Name | Select-Object -Last 1
            $meta = if ($s) { Join-Path $s.FullName 'session.json' }
            if ($meta -and (Test-Path $meta)) {
                $url = (Get-Content -Raw $meta | ConvertFrom-Json).dashboard
                if ($url) {
                    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 "$($url)api/live" | Out-Null; Start-Process $url; return } catch {}
                }
            }
            Start-Sleep -Milliseconds 500
        }
    } | Out-Null
}

Write-Host "elrobot-sim: Claude operator (robot tools only). Recordings: $SimRecordDir"
Push-Location $work
try {
    & claude '--tools=' --strict-mcp-config --mcp-config $config --allowedTools 'mcp__elrobot-sim' `
        --append-system-prompt-file (Join-Path $here 'OPERATOR.md') @ClaudeArgs
} finally {
    Pop-Location
    Remove-Item -Force $config -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2   # let the server stop; Claude kills it, so build the replay here
    $session = Get-ChildItem -Path $SimRecordDir -Directory | Sort-Object Name | Select-Object -Last 1
    if ($session) {
        Write-Host "building replay for $($session.Name) ..."
        $replayCmd = if ($SimRunSh) { @('bash', $SimRunSh, 'sim-replay') } else { @('python3', "$(To-Wsl $here)/make_replay.py") }
        $replayArgs = @('-d', $SimDistro, '--') + $replayCmd + @((To-Wsl $session.FullName))
        & wsl.exe @replayArgs | Out-Null
        $replay = Join-Path $session.FullName 'replay.html'
        if (Test-Path $replay) {
            Write-Host "replay: $replay"
            if (-not $SimNoBrowser) { Start-Process $replay }
        }
    }
}
