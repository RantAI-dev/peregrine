#!/bin/sh
# Install the `peregrine` CLI from a GitHub release.
#
#   curl -fsSL https://peregrine.rantai.dev/install.sh | sh
#   curl -fsSL https://peregrine.rantai.dev/install.sh | sh -s -- --prefix /usr/local
#
# peregrine.rantai.dev redirects to this script in the newest GitHub release (the lakehouse
# pattern); the GitHub URL is the same file: .../RantAI-dev/peregrine/main/scripts/install.sh.
#
# The archive is verified against the release's SHA256SUMS before anything is installed, and the
# binary replaces the old one with an atomic rename, so content-addressed or running binaries are
# never truncated. Linux x86-64 and aarch64 have archives today; macOS on Apple Silicon ships from
# the release after v0.2.0. The released binaries are portable per architecture and pick their
# AVX2 / AVX-512 / NEON kernels at run time (`peregrine tier`). Intel Macs build from source.
#
# Options:
#   --prefix DIR      install to DIR/bin (default: $PEREGRINE_PREFIX, else ~/.local)
#   --version TAG     a release tag such as v0.2.0 (default: $PEREGRINE_VERSION, else the newest)
#   -h, --help        print this help
#
# Environment:
#   PEREGRINE_REPO      owner/repo to download from (default: RantAI-dev/peregrine)
#   PEREGRINE_BASE_URL  override the release download base (for mirrors and tests)
#   PEREGRINE_INSTALL_TEST_OS / PEREGRINE_INSTALL_TEST_MACHINE
#                       replace `uname -s` / `uname -m`, for scripts/install-test.sh only
set -eu

repo="${PEREGRINE_REPO:-RantAI-dev/peregrine}"
prefix="${PEREGRINE_PREFIX:-$HOME/.local}"
version="${PEREGRINE_VERSION:-latest}"

usage() {
    cat <<'EOF'
Install the `peregrine` CLI from a GitHub release.

  curl -fsSL https://peregrine.rantai.dev/install.sh | sh
  curl -fsSL https://peregrine.rantai.dev/install.sh | sh -s -- --prefix /usr/local

The archive is verified against the release's SHA256SUMS before anything is installed, and the
binary replaces the old one with an atomic rename, so content-addressed or running binaries are
never truncated. Linux x86-64 and aarch64 have archives today; macOS on Apple Silicon ships from
the release after v0.2.0. The released binaries are portable per architecture and pick their
AVX2 / AVX-512 / NEON kernels at run time (`peregrine tier`). Intel Macs build from source.

On macOS, from the release after v0.2.0, a binary downloaded with a browser carries the quarantine
flag and Gatekeeper blocks it; one fetched by this script with curl does not. If you used a
browser, run `xattr -d com.apple.quarantine ~/.local/bin/peregrine` once.

Options:
  --prefix DIR      install to DIR/bin (default: $PEREGRINE_PREFIX, else ~/.local)
  --version TAG     a release tag such as v0.2.0 (default: $PEREGRINE_VERSION, else the newest)
  -h, --help        print this help

Environment:
  PEREGRINE_REPO      owner/repo to download from (default: RantAI-dev/peregrine)
  PEREGRINE_BASE_URL  override the release download base (for mirrors and tests)
EOF
    exit "${1:-0}"
}

die() {
    printf 'peregrine-install: %s\n' "$*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
    --prefix)
        [ $# -ge 2 ] || die "--prefix needs a directory"
        prefix="$2"
        shift 2
        ;;
    --prefix=*)
        prefix="${1#*=}"
        shift
        ;;
    --version)
        [ $# -ge 2 ] || die "--version needs a tag"
        version="$2"
        shift 2
        ;;
    --version=*)
        version="${1#*=}"
        shift
        ;;
    -h | --help)
        usage 0
        ;;
    *)
        die "unknown argument: $1 (try --help)"
        ;;
    esac
done

# Expand a leading ~ from --prefix=~/x, which the shell cannot expand after the =.
case "$prefix" in
"~") prefix="$HOME" ;;
"~/"*) prefix="$HOME/${prefix#"~/"}" ;;
esac

os="${PEREGRINE_INSTALL_TEST_OS:-$(uname -s)}"
machine="${PEREGRINE_INSTALL_TEST_MACHINE:-$(uname -m)}"

case "$os" in
Linux | Darwin) ;;
*) die "release archives are for Linux and macOS on Apple Silicon; on $os build from source: cargo build --release" ;;
esac

case "$machine" in
x86_64 | amd64) arch=x86_64 ;;
aarch64 | arm64) arch=aarch64 ;;
*) die "no release archive for $os on $machine; build from source: cargo build --release" ;;
esac

case "$os-$arch" in
Linux-x86_64) short=linux-x86_64 ;;
Linux-aarch64) short=linux-aarch64 ;;
Darwin-aarch64) short=macos-aarch64 ;;
Darwin-x86_64) die "macOS releases are Apple Silicon (arm64) only; Intel Macs build from source: cargo build --release" ;;
*) die "no release archive for $os on $machine; build from source: cargo build --release" ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar >/dev/null 2>&1 || die "tar is required"
if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    die "sha256sum or shasum is required"
fi

if [ "$version" = latest ]; then
    version="$(
        curl -fsSL "https://api.github.com/repos/$repo/releases/latest" |
            sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p'
    )"
    [ -n "$version" ] || die "could not read the newest release tag; pass --version vX.Y.Z"
fi

asset="peregrine-$version-$short.tar.gz"
base="${PEREGRINE_BASE_URL:-https://github.com/$repo/releases/download/$version}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

printf 'Downloading %s\n' "$asset"
if ! curl -fsSL -o "$tmp/$asset" "$base/$asset"; then
    # v0.2.0 has no macOS archive, and a Mac that asks for one deserves to be told that rather
    # than shown curl's 404: the archive and this line ship from the release after v0.2.0.
    case "$os" in
    Darwin)
        die "no macOS archive in release $version (or the download failed): macOS on Apple Silicon ships from the release after v0.2.0; build from source (cargo build --release) or wait for the next release"
        ;;
    esac
    die "download failed: $base/$asset"
fi
curl -fsSL -o "$tmp/SHA256SUMS" "$base/SHA256SUMS" || die "download failed: $base/SHA256SUMS"

expected="$(
    awk -v f="$asset" '{
        n = $2
        sub(/^\*/, "", n); sub(/^\.\//, "", n); sub(/^.*\//, "", n)
        if (n == f) { print $1; exit }
    }' "$tmp/SHA256SUMS"
)"
[ -n "$expected" ] || die "SHA256SUMS has no entry for $asset"
actual="$(sha256 "$tmp/$asset")"
[ "$actual" = "$expected" ] || die "checksum mismatch for $asset: expected $expected, got $actual"

tar -xzf "$tmp/$asset" -C "$tmp"
staged="$tmp/peregrine-$version-$short/peregrine"
[ -x "$staged" ] || die "archive did not contain peregrine-$version-$short/peregrine"

bindir="$prefix/bin"
mkdir -p "$bindir"

# Rename into place: a running server keeps its old inode instead of failing with ETXTBSY.
cp "$staged" "$bindir/.peregrine.$$"
chmod 0755 "$bindir/.peregrine.$$"
mv -f "$bindir/.peregrine.$$" "$bindir/peregrine"

printf 'Installed %s\n' "$bindir/peregrine"
"$bindir/peregrine" --version

case ":$PATH:" in
*":$bindir:"*) ;;
*)
    printf '\n%s is not on PATH. Add it to your shell profile:\n' "$bindir"
    printf '  export PATH="%s:$PATH"\n' "$bindir"
    ;;
esac

printf '\nNext: peregrine tier          # the kernel tier picked for this CPU\n'
