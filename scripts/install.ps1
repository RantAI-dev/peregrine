# Install the `peregrine` CLI from a GitHub release on Windows x86-64.
#
#   irm https://peregrine.rantai.dev/install.ps1 | iex
#   irm https://peregrine.rantai.dev/install.ps1 -OutFile install.ps1; .\install.ps1 -Version v0.2.0
#
# The vanity host redirects to this script in the newest GitHub release (the same lakehouse
# pattern as install.sh); the GitHub URL is the same file:
# .../RantAI-dev/peregrine/main/scripts/install.ps1.
#
# The zip is verified against the release's SHA256SUMS with Get-FileHash before anything is
# installed. It installs to %LOCALAPPDATA%\Programs\peregrine (no admin rights), adds that
# directory to the *user* Path when it is missing, and says how to undo both. A running
# peregrine.exe cannot be overwritten on Windows, so the old one is renamed to
# peregrine.exe.old first and the next start removes it.
#
# Written for Windows PowerShell 5.1, the version every supported Windows ships.
#
# Parameters:
#   -Version TAG      a release tag such as v0.2.0 (default: the newest release)
#   -InstallDir DIR   install to DIR (default: %LOCALAPPDATA%\Programs\peregrine)
#   -BaseUrl URL|DIR  release download base: a mirror, or a local directory in tests
#   -Repo OWNER/REPO  GitHub repository (default: RantAI-dev/peregrine)
#   -Help             print this help
[CmdletBinding()]
param(
    [string]$Version = "",
    [string]$InstallDir = "",
    [string]$BaseUrl = "",
    [string]$Repo = "RantAI-dev/peregrine",
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Show-Help {
    @'
Install the peregrine CLI from a GitHub release.

  irm https://peregrine.rantai.dev/install.ps1 | iex
  irm https://peregrine.rantai.dev/install.ps1 -OutFile install.ps1; .\install.ps1 -Version v0.2.0
  .\install.ps1 -Version v0.2.0 -InstallDir C:\tools\peregrine

The zip is verified against the release's SHA256SUMS with Get-FileHash. It installs to
%LOCALAPPDATA%\Programs\peregrine and adds that directory to your user Path (never the machine
Path); no administrator rights are needed. Windows x86-64 only: there is no ARM64 archive yet.

Parameters:
  -Version TAG      a release tag such as v0.2.0 (default: the newest release)
  -InstallDir DIR   install to DIR (default: %LOCALAPPDATA%\Programs\peregrine)
  -BaseUrl URL|DIR  release download base: a mirror, or a local directory (for tests)
  -Repo OWNER/REPO  GitHub repository (default: RantAI-dev/peregrine)
  -Help             print this help
'@
}

if ($Help) {
    Show-Help
    exit 0
}

# `PROCESSOR_ARCHITECTURE` is the architecture of this process; in a 32-bit shell on a 64-bit
# machine the real machine is in PROCESSOR_ARCHITEW6432.
$arch = $env:PROCESSOR_ARCHITECTURE
if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
if ($arch -eq "ARM64") {
    throw "Peregrine has no Windows ARM64 release yet (x86-64 only); build from source: cargo build --release"
}
if ($arch -ne "AMD64" -and $arch -ne "x86_64") {
    throw "no release archive for Windows on $arch; build from source: cargo build --release"
}

if (-not $InstallDir) {
    $local = $env:LOCALAPPDATA
    if (-not $local) { $local = [Environment]::GetFolderPath("LocalApplicationData") }
    if (-not $local) { throw "LOCALAPPDATA is not set; pass -InstallDir DIR" }
    $InstallDir = Join-Path $local "Programs\peregrine"
}
$InstallDir = [IO.Path]::GetFullPath($InstallDir)

# Windows PowerShell 5.1 defaults to SSL 3.0/TLS 1.0 on older builds; GitHub needs TLS 1.2+.
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
    # A newer .NET may already have a better default; the download decides.
}

function Save-Remote([string]$From, [string]$To) {
    if ($From -match "^https?://") {
        Invoke-WebRequest -UseBasicParsing -Uri $From -OutFile $To
    } elseif ($From -match "^file://") {
        Copy-Item -LiteralPath ([uri]$From).LocalPath -Destination $To -Force
    } else {
        Copy-Item -LiteralPath $From -Destination $To -Force
    }
}

if (-not $Version) {
    $release = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repo/releases/latest"
    $Version = $release.tag_name
    if (-not $Version) {
        throw "could not read the newest release tag; pass -Version vX.Y.Z"
    }
}

$asset = "peregrine-$Version-windows-x86_64.zip"
if ($BaseUrl) {
    $base = $BaseUrl.TrimEnd("/", "\")
} else {
    $base = "https://github.com/$Repo/releases/download/$Version"
}
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("peregrine-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
    $zip = Join-Path $tmp $asset
    $sums = Join-Path $tmp "SHA256SUMS"

    Write-Host "Downloading $asset"
    Save-Remote "$base/$asset" $zip
    Save-Remote "$base/SHA256SUMS" $sums

    $expected = $null
    foreach ($line in Get-Content -LiteralPath $sums) {
        if ($line -match "^\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$") {
            $name = ($Matches[2] -split "[\\/]")[-1]
            if ($name -eq $asset) { $expected = $Matches[1].ToLowerInvariant(); break }
        }
    }
    if (-not $expected) { throw "SHA256SUMS has no entry for $asset" }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "checksum mismatch for ${asset}: expected $expected, got $actual"
    }
    Write-Host "Verified SHA-256 $actual"

    Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
    $staged = Join-Path $tmp ("peregrine-$Version-windows-x86_64")
    $stagedExe = Join-Path $staged "peregrine.exe"
    if (-not (Test-Path -LiteralPath $stagedExe)) {
        # Tolerate an archive whose folder is named differently; the zip is already verified.
        $found = Get-ChildItem -LiteralPath $tmp -Recurse -Filter "peregrine.exe" |
            Select-Object -First 1
        if (-not $found) { throw "the archive has no peregrine.exe" }
        $stagedExe = $found.FullName
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $target = Join-Path $InstallDir "peregrine.exe"
    if (Test-Path -LiteralPath $target) {
        try {
            Copy-Item -LiteralPath $stagedExe -Destination $target -Force
        } catch {
            # The running .exe can be renamed but not replaced: move it aside and retry.
            $old = "$target.old"
            Remove-Item -LiteralPath $old -Force -ErrorAction SilentlyContinue
            Move-Item -LiteralPath $target -Destination $old -Force
            Copy-Item -LiteralPath $stagedExe -Destination $target -Force
            Write-Host "The previous peregrine.exe was running; it is $old and goes away on the next start."
        }
    } else {
        Copy-Item -LiteralPath $stagedExe -Destination $target -Force
    }

    # User scope only: never the machine Path, and no admin rights.
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $entries = @()
    if ($userPath) { $entries = @($userPath -split ";" | Where-Object { $_ -ne "" }) }
    $have = @($entries | Where-Object { $_.TrimEnd("\") -ieq $InstallDir.TrimEnd("\") })
    if ($have.Count -eq 0) {
        $newPath = (@($entries) + $InstallDir) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "Added $InstallDir to your user PATH (open a new terminal to pick it up)."
    } else {
        Write-Host "$InstallDir is already on your user PATH."
    }

    Write-Host ""
    Write-Host "Installed $target"
    & $target --version

    Write-Host ""
    Write-Host "Next: peregrine tier          # the kernel tier picked for this CPU"
    Write-Host ""
    Write-Host "To undo:"
    Write-Host "  Remove-Item -Recurse -Force '$InstallDir'"
    Write-Host "  # then remove '$InstallDir' from your user Path:"
    Write-Host "  #   Settings > System > About > Advanced system settings > Environment Variables"
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
