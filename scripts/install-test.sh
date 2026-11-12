#!/bin/sh
# Exercise scripts/install.sh without a network release: build stub archives for each platform,
# point the installer at the directory with PEREGRINE_BASE_URL and check what it installs or
# refuses. Runs on Linux and macOS, and needs no Rust build.
#
#   scripts/install-test.sh
#
# The platform-detection overrides (PEREGRINE_INSTALL_TEST_OS / _MACHINE) exist for this test: on
# a Linux host they also exercise the Darwin arm64 path and the Intel refusal. The real archive is
# installed by the release workflow and .github/workflows/macos-dev.yml.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/Cargo.toml" | head -1)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

release="$tmp/release"
mkdir -p "$release"

# One stub archive per platform, with the same layout build.sh produces. The binary is a shell
# script so it runs whatever host is testing the Darwin path.
stub() {
    short=$1
    stage="$release/peregrine-v$version-$short"
    mkdir -p "$stage"
    cat > "$stage/peregrine" <<'EOF'
#!/bin/sh
case "${1:-}" in
--version) echo "peregrine 0.0.0 (install-test)" ;;
tier) echo "kernel tier   aarch64-neon-dotprod (install-test)" ;;
*) echo "peregrine (install-test)" ;;
esac
EOF
    chmod 755 "$stage/peregrine"
    tar -czf "$release/peregrine-v$version-$short.tar.gz" -C "$release" "peregrine-v$version-$short"
    rm -rf "$stage"
    (cd "$release" && printf '%s  %s\n' "$(sha256 "peregrine-v$version-$short.tar.gz")" \
        "peregrine-v$version-$short.tar.gz" >> SHA256SUMS)
}

for short in linux-x86_64 linux-aarch64 macos-aarch64; do
    stub "$short"
done

fail=0
pass() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1" >&2; fail=1; }

# One case: OS, machine, expected asset (or "refuse"), and a phrase the refusal must contain.
try() {
    os=$1
    machine=$2
    want=$3
    phrase=$4
    out="$tmp/out-$os-$machine"
    prefix="$tmp/prefix-$os-$machine"
    set +e
    PEREGRINE_BASE_URL="file://$release" \
        PEREGRINE_INSTALL_TEST_OS="$os" \
        PEREGRINE_INSTALL_TEST_MACHINE="$machine" \
        sh "$root/scripts/install.sh" --version "v$version" --prefix "$prefix" \
        >"$out" 2>&1
    status=$?
    set -e
    if [ "$want" = refuse ]; then
        if [ "$status" -ne 0 ] && grep -F -q "$phrase" "$out"; then
            pass "$os/$machine is refused: $phrase"
        else
            bad "$os/$machine was not refused with '$phrase'"
            sed 's/^/    /' "$out" >&2
        fi
        return
    fi
    if [ "$status" -ne 0 ]; then
        bad "$os/$machine install failed"
        sed 's/^/    /' "$out" >&2
        return
    fi
    if [ -x "$prefix/bin/peregrine" ] &&
        "$prefix/bin/peregrine" --version | grep -q 'install-test' &&
        "$prefix/bin/peregrine" tier | grep -q 'kernel tier' &&
        grep -F -q "peregrine-v$version-$want.tar.gz" "$out"; then
        pass "$os/$machine installs $want"
    else
        bad "$os/$machine did not install $want"
        sed 's/^/    /' "$out" >&2
    fi
}

try Linux x86_64 linux-x86_64 ""
try Linux aarch64 linux-aarch64 ""
try Darwin arm64 macos-aarch64 ""
try Darwin aarch64 macos-aarch64 ""
try Darwin x86_64 refuse "Apple Silicon (arm64) only"
try FreeBSD amd64 refuse "build from source"
try Linux riscv64 refuse "build from source"

if [ "$fail" -ne 0 ]; then
    exit 1
fi
printf 'install-test: all cases passed on %s/%s\n' "$(uname -s)" "$(uname -m)"
