# Claude Code on Windows (logged in) as the operator of the REAL arm; the elrobot-real MCP server runs in WSL
# over wsl.exe stdio. Same restrictions as the sim: only the elrobot-real tools, no shell/file tools, empty cwd.
# A human must watch the arm and be ready to stop it (Ctrl-C here; the servos then hold their last goal).
#   powershell -ExecutionPolicy Bypass -File sim\run_agent_real_windows.ps1 [-RealDistro Ubuntu]
#       [-RealRunSh /mnt/d/.../run.sh] [-RealNoBrowser] [claude args, e.g. --model sonnet -p "Start."]
# Arguments are parsed by hand: a param block would let PowerShell prefix-match Claude flags like -p.
$RealDistro = 'Ubuntu'; $RealRunSh = ''; $RealNoBrowser = $false; $ClaudeArgs = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    switch -CaseSensitive ($args[$i]) {
        '-RealDistro'    { $RealDistro = $args[++$i] }
        '-RealRunSh'     { $RealRunSh = $args[++$i] }      # WSL path of a run.sh that knows 'real-arm-mcp' (machine config)
        '-RealNoBrowser' { $RealNoBrowser = $true }
        default          { $ClaudeArgs += [string]$args[$i] }
    }
}
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
function To-Wsl([string]$p) { (wsl.exe -d $RealDistro -- wslpath -a ($p -replace '\\', '/')).Trim() }
$server = if ($RealRunSh) { @('bash', $RealRunSh, 'real-arm-mcp') } else { @('bash', "$(To-Wsl $here)/run_real_server.sh") }
$config = Join-Path $env:TEMP "elrobot-real-mcp-$PID.json"
@{ mcpServers = @{ 'elrobot-real' = @{ command = 'wsl.exe'; args = @('-d', $RealDistro, '--') + $server } } } |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding ascii $config
$work = Join-Path $env:TEMP "elrobot-real-agent-$PID"
New-Item -ItemType Directory -Force -Path $work | Out-Null
if (-not $RealNoBrowser) { Start-Process 'http://localhost:8890/' }   # live camera relay (cam_stream)
Write-Host "elrobot-real: Claude operating the REAL arm. Ctrl-C to stop. Live cameras: http://localhost:8890/"
Push-Location $work
try {
    & claude '--tools=' --strict-mcp-config --mcp-config $config --allowedTools 'mcp__elrobot-real' `
        --append-system-prompt-file (Join-Path $here 'OPERATOR_REAL.md') @ClaudeArgs
} finally {
    Pop-Location
    Remove-Item -Force $config -ErrorAction SilentlyContinue
}
