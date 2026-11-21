# Claude Code on Windows (already logged in) as System 2; the System 2 server, the simulator and
# Open-Jev run inside WSL. Claude gets ONLY the elrobot-system2 tools.
#   powershell -ExecutionPolicy Bypass -File system2\run_agent_windows.ps1 [-Policy geometric|jev|shadow]
#       [-Distro Ubuntu] [-RecordDir D:\x] [-NoGui] [claude args...]      e.g. -p "Put the cube in the box."
# Arguments are parsed by hand so PowerShell does not prefix-match Claude flags such as -p.
$Policy = 'geometric'; $Distro = 'Ubuntu'; $RecordDir = ''; $Gui = '1'
$ClaudeArgs = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    switch -CaseSensitive ($args[$i]) {
        '-Policy'    { $Policy = $args[++$i] }
        '-Distro'    { $Distro = $args[++$i] }
        '-RecordDir' { $RecordDir = $args[++$i] }
        '-NoGui'     { $Gui = '0' }
        default      { $ClaudeArgs += [string]$args[$i] }
    }
}
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$repo = Split-Path $here -Parent
if (-not $RecordDir) { $RecordDir = Join-Path $repo 'sim\sessions' }
New-Item -ItemType Directory -Force -Path $RecordDir | Out-Null
function To-Wsl([string]$p) { (wsl.exe -d $Distro -- wslpath -a ($p -replace '\\', '/')).Trim() }
$wslHere = To-Wsl $here

if ($Policy -ne 'geometric') {
    & wsl.exe -d $Distro -- bash "$wslHere/open_jev.sh" --background
    if ($LASTEXITCODE -ne 0) { throw 'Open-Jev did not start (see /tmp/open-jev.log in WSL)' }
}
$wslArgs = @('-d', $Distro, '--', 'env', "ROBOT_SIM_GUI=$Gui", "ROBOT_SIM_RECORD_DIR=$(To-Wsl $RecordDir)",
             "S2_POLICY=$Policy", 'bash', "$wslHere/run_server.sh")
$config = Join-Path $env:TEMP "elrobot-system2-mcp-$PID.json"
@{ mcpServers = @{ 'elrobot-system2' = @{ command = 'wsl.exe'; args = $wslArgs } } } | ConvertTo-Json -Depth 5 |
    Set-Content -Encoding ascii $config
$work = Join-Path $env:TEMP "elrobot-system2-agent-$PID"
New-Item -ItemType Directory -Force -Path $work | Out-Null
if (-not $env:MCP_TIMEOUT) { $env:MCP_TIMEOUT = '120000' }  # first start builds the scene and loads MuJoCo

Write-Host "elrobot System 2 (policy: $Policy). Live dashboard: http://localhost:8891  Recordings: $RecordDir"
Push-Location $work
try {
    & claude '--tools=' --strict-mcp-config --mcp-config $config --allowedTools 'mcp__elrobot-system2' `
        --append-system-prompt-file (Join-Path $here 'PLANNER.md') @ClaudeArgs
} finally {
    Pop-Location
    Remove-Item -Force $config -ErrorAction SilentlyContinue
}
