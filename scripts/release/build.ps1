# Build the Windows release archive for one target (docs/dev/releasing.md).
#
#   scripts/release/build.ps1 x86_64-pc-windows-msvc   # on an x86-64 Windows host or runner
#
# This is the PowerShell twin of scripts/release/build.sh. The Windows runner's native archive
# tool is Compress-Archive and it has no `strip` or `sha256sum`; rather than make the bash script
# depend on the extras Git Bash happens to carry, each system keeps one small script. The build is
# native (there is no cross-compilation), and it is exactly what Windows PowerShell 5.1 runs.
#
# The MSVC build links the C runtime statically (-C target-feature=+crt-static), so the published
# .exe needs no Visual C++ redistributable. That is a release-packaging flag only: the binary still
# has no target-cpu and picks its AVX2 / AVX-512 kernels at run time (`peregrine tier` prints the
# tier it picked).
#
# The output is
#   dist/peregrine-v<version>-windows-x86_64.zip   (peregrine.exe, LICENSE, NOTICE)
#   dist/SHA256SUMS                               (one line for that archive)
# and the release job concatenates the targets' SHA256SUMS files into the release's.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

if ($args.Count -ne 1) {
    Write-Error "usage: scripts/release/build.ps1 <target-triple>  (build one target, natively)"
    exit 1
}
$target = $args[0]

$hostTriple = (& rustc -vV | Select-String -Pattern "^host: ").ToString().Substring(6).Trim()
if ($target -ne $hostTriple) {
    Write-Error "scripts/release: target $target is not this host ($hostTriple); release targets are built natively on their own runners"
    exit 1
}

$short = switch ($target) {
    "x86_64-pc-windows-msvc" { "windows-x86_64" }
    default { $target }
}
$version = (Select-String -Path "Cargo.toml" -Pattern '^version = "(.+)"$' |
    Select-Object -First 1).Matches.Groups[1].Value
$name = "peregrine-v$version-$short"

Write-Host "scripts/release: building $name (host $hostTriple, version $version)"
$env:RUSTFLAGS = "-C target-feature=+crt-static"
& cargo build --release --locked -p peregrine --target $target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$bin = Join-Path "target" (Join-Path $target "release/peregrine.exe")
if (-not (Test-Path -LiteralPath $bin)) {
    Write-Error "scripts/release: $bin was not built"
    exit 1
}
# Informational: what the portable binary picks on this machine (an avx2 tier here is the
# run-time dispatch working; the release workflow asserts it).
& $bin tier | Select-Object -First 1

$dist = "dist"
$stage = Join-Path $dist $name
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -LiteralPath $bin -Destination (Join-Path $stage "peregrine.exe")
Copy-Item -LiteralPath "LICENSE" -Destination $stage
Copy-Item -LiteralPath "NOTICE" -Destination $stage

$zip = Join-Path $dist "$name.zip"
Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
# Write the entries by hand with `/` separators: Compress-Archive uses `\` on Windows, which a
# POSIX unzip reads as part of the file name instead of a directory separator.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipStream = [IO.File]::Open($zip, [IO.FileMode]::Create)
try {
    $archive = New-Object IO.Compression.ZipArchive($zipStream, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($file in Get-ChildItem -LiteralPath $stage -File) {
            $entry = "$name/$($file.Name)"
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $file.FullName, $entry,
                [IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally {
        $archive.Dispose()
    }
} finally {
    $zipStream.Dispose()
}
Remove-Item -LiteralPath $stage -Recurse -Force

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
Set-Content -LiteralPath (Join-Path $dist "SHA256SUMS") -Value "$hash  $name.zip" -Encoding ASCII
Write-Host "scripts/release: wrote $zip and $dist/SHA256SUMS"
