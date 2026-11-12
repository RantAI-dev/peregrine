#!/usr/bin/env bash
# Build the release archives for one target (docs/dev/releasing.md).
#
#   scripts/release/build.sh x86_64-unknown-linux-gnu    # on an x86-64 host or runner
#   scripts/release/build.sh aarch64-unknown-linux-gnu   # on an aarch64 host or runner
#   scripts/release/build.sh aarch64-apple-darwin        # on an Apple Silicon Mac or runner
#
# Each target is built natively on its own host (the release workflow runs one job per runner);
# there is no cross-compilation. The output is
# dist/peregrine-v<version>-linux-x86_64.tar.gz (or -linux-aarch64, -macos-aarch64) with the
# stripped `peregrine` binary, LICENSE and NOTICE, plus a one-line dist/SHA256SUMS covering that
# archive; the release job concatenates the targets' files into the release's SHA256SUMS.
#
# The build is portable by construction: the x86-64 binary compiles for the plain x86-64
# baseline (no RUSTFLAGS, no target-cpu=native) and picks its AVX2 / AVX-512 kernels at run
# time (crates/peregrine-kernels/src/cpu.rs). `./peregrine tier` prints the tier it picked.
#
# macOS: the deployment target is 11.0 (the first Apple Silicon macOS), the binary is stripped
# with `strip -x` and then ad-hoc signed with `codesign -s -`, which the linker already did but
# `strip` invalidates. There is no notarisation (docs/quickstart.md says what a browser download
# needs). `sha256sum` does not exist there: `shasum -a 256` writes the same SHA256SUMS line.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -n "${RUSTFLAGS:-}" ]]; then
    echo "scripts/release: refusing to build with RUSTFLAGS set ('${RUSTFLAGS}'): release" \
        "artefacts must keep the portable baseline and the run-time ISA dispatch" >&2
    exit 1
fi

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <target-triple>  (build one target, natively)" >&2
    exit 1
fi
target="$1"

# The archive name is the friendly one users download: linux-x86_64 / linux-aarch64 / macos-aarch64.
case "$target" in
    x86_64-unknown-linux-gnu) short="linux-x86_64" ;;
    aarch64-unknown-linux-gnu) short="linux-aarch64" ;;
    aarch64-apple-darwin) short="macos-aarch64" ;;
    *) short="$target" ;;
esac

host="$(rustc -vV | sed -n 's/^host: //p')"
if [[ "$target" != "$host" ]]; then
    echo "scripts/release: target $target is not this host ($host); release targets are" \
        "built natively on their own runners" >&2
    exit 1
fi

# The oldest macOS an archive runs on. Set for the build (the runner's SDK is newer).
if [[ "$target" == *-apple-darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
fi

version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)"
dist="dist"
name="peregrine-v${version}-${short}"
mkdir -p "$dist"

echo "scripts/release: building $name (host $host, version $version)"
cargo build --release --locked -p peregrine --target "$target"

bin="${CARGO_TARGET_DIR:-target}/${target}/release/peregrine"
# Informational: what the portable binary picks on this machine (x86-64: a baseline build
# reporting an avx2 tier here is the run-time dispatch working; the release workflow asserts it).
"$bin" tier | sed -n '1p'

stage="$dist/$name"
rm -rf "$stage"
mkdir -p "$stage"
cp "$bin" "$stage/peregrine"
if [[ "$target" == *-apple-darwin ]]; then
    # `strip` invalidates the linker's ad-hoc signature, so sign the stripped file again.
    strip -x "$stage/peregrine"
    codesign --force --sign - "$stage/peregrine"
    codesign --verify --strict "$stage/peregrine"
    codesign -dv "$stage/peregrine" 2>&1 | grep -E '^(Identifier|Format|Signature)='
    if ! codesign -dv "$stage/peregrine" 2>&1 | grep -q '^Signature=adhoc'; then
        echo "scripts/release: $stage/peregrine is not ad-hoc signed" >&2
        exit 1
    fi
else
    strip "$stage/peregrine"
fi
chmod 755 "$stage/peregrine"
cp LICENSE NOTICE "$stage/"

tar -czf "$dist/$name.tar.gz" -C "$dist" "$name"
rm -rf "$stage"

if command -v sha256sum >/dev/null 2>&1; then
    (cd "$dist" && sha256sum "$name.tar.gz" > SHA256SUMS)
else
    (cd "$dist" && shasum -a 256 "$name.tar.gz" > SHA256SUMS)
fi
echo "scripts/release: wrote $dist/$name.tar.gz and $dist/SHA256SUMS"
