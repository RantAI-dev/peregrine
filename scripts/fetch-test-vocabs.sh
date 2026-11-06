#!/usr/bin/env bash
# Fetch tokenizer vocabulary test files from llama.cpp v0.9.0
set -euo pipefail

PINNED_COMMIT="cd959bae99ff4bed22bef2604f45241c09bbb8c8"
BASE_URL="https://raw.githubusercontent.com/ggml-org/llama.cpp/${PINNED_COMMIT}/models"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/models/vocab"

mkdir -p "${TARGET_DIR}"

FILES=(
    "ggml-vocab-llama-spm.gguf"
    "ggml-vocab-llama-spm.gguf.inp"
    "ggml-vocab-llama-spm.gguf.out"
)

# Pinned SHA-256 checksums
CHECKSUMS="
16c3724582d59aa8bf84711894e833f916ee46a31d80e21312759c48bf8d0e69  ${TARGET_DIR}/ggml-vocab-llama-spm.gguf
be0a11f7071f0c67d3053a2d377d8f35f0dbcb77ff11a60300fb57d36b477cf0  ${TARGET_DIR}/ggml-vocab-llama-spm.gguf.inp
ad6905c925c49c022974fe67e030382e8a4f56eb4489a0b9481dcedb1ec73b29  ${TARGET_DIR}/ggml-vocab-llama-spm.gguf.out
"

for file in "${FILES[@]}"; do
    dest="${TARGET_DIR}/${file}"
    if [[ ! -f "${dest}" ]]; then
        echo "Fetching ${file}..."
        curl -sSfL "${BASE_URL}/${file}" -o "${dest}"
    else
        echo "Found existing ${file}"
    fi
done

echo "Verifying SHA-256 checksums..."
echo "${CHECKSUMS}" | sha256sum --check --status

echo "All vocabulary test files verified successfully in ${TARGET_DIR}."
