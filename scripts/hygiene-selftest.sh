#!/usr/bin/env bash
# Self-test for scripts/hygiene.sh. Every credential-shaped string is assembled
# at run time from pieces, so this file itself carries none.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
hygiene="$here/hygiene.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hygiene-selftest.XXXXXX")
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT HUP INT TERM

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

ok() { printf 'ok: %s\n' "$*"; }

cd "$tmp"
git init -q
git config user.email selftest@example.invalid
git config user.name selftest
git config commit.gpgsign false
printf 'clean\n' > README
git add README
git commit -q -m base

out="$tmp/hygiene.out"

expect_pass() {
  local label=$1
  shift
  if ! "$hygiene" "$@" > "$out" 2>&1; then
    cat "$out" >&2
    fail "$label: hygiene failed on a clean tree"
  fi
  ok "$label"
}

expect_fail() {
  local label=$1
  shift
  if "$hygiene" "$@" > "$out" 2>&1; then
    fail "$label: hygiene passed but should have failed"
  fi
  ok "$label"
}

# Also assert that the matched text stays out of the output.
expect_fail_secret() {
  local label=$1 secret=$2 content=$3
  printf '%s\n' "$content" > leak.txt
  git add leak.txt
  if "$hygiene" > "$out" 2>&1; then
    fail "$label: hygiene passed but should have failed"
  fi
  if grep -F -- "$secret" "$out" >/dev/null; then
    fail "$label: hygiene printed the matched text"
  fi
  git rm -q --cached leak.txt
  rm -f leak.txt
  ok "$label"
}

add_path() {
  printf 'not a secret\n' > "$1"
  git add -- "$1"
}

remove_path() {
  git rm -q --cached -- "$1" 2>/dev/null || true
  rm -f -- "$1"
}

# Clean tree, including an empty .env.example.
expect_pass 'clean tree'

printf '# sample\nTOKEN=\nPASSWORD=\n' > .env.example
git add .env.example
expect_pass '.env.example with no values'
remove_path .env.example

# Credential-style paths.
for path in .env .env.local server.pem tls.key store.p12 bundle.pfx id_rsa \
  id_ed25519 .netrc .npmrc .pypirc credentials.json prod.tfstate; do
  add_path "$path"
  expect_fail "credential path: $path"
  remove_path "$path"
done

# A .env.example that actually carries a value is not allowed.
val='hv''alue123'
printf 'TOKEN=%s\n' "$val" > .env.example
git add .env.example
expect_fail '.env.example with a value'
remove_path .env.example

# Private-key header.
header='-----BE''GIN RSA PRIVATE KEY-----'
expect_fail_secret 'private-key header' "$header" "$header"

# Assignments. Each key and value is split so the source has no credential.
expect_fail_secret 'PASSWORD assignment' 'hun''ter2' "PASS""WORD=hun""ter2"
expect_fail_secret 'SSH_PASS assignment' 'hun''ter2' "SSH_""PA""SS=hun""ter2"
expect_fail_secret 'DB_PASSWORD assignment' 'hun''ter2' "DB_""PASS""WORD=hun""ter2"
expect_fail_secret 'GITHUB_TOKEN assignment' 'hun''ter2' "GITHUB_""TO""KEN=hun""ter2"
expect_fail_secret 'API_KEY assignment' 'hun''ter2' "API_""KEY=hun""ter2"
expect_fail_secret 'lowercase password assignment' 'hun''ter2' "pass""word=hun""ter2"

# Well-known token shapes.
gh_prefix='gh''p_'
gh_body='A1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8'
expect_fail_secret 'GitHub token' "${gh_prefix}${gh_body}" "${gh_prefix}${gh_body}"

pat_prefix='github_''pat_'
pat_body='A1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6'
expect_fail_secret 'GitHub fine-grained token' "${pat_prefix}${pat_body}" "${pat_prefix}${pat_body}"

sk_prefix='s''k-'
sk_body='A1B2C3D4E5F6G7H8I9J0K1L2'
expect_fail_secret 'API key token' "${sk_prefix}${sk_body}" "${sk_prefix}${sk_body}"

aws_prefix='AK''IA'
aws_body='ABCDEFGHIJKLMNOP'
expect_fail_secret 'AWS access key' "${aws_prefix}${aws_body}" "${aws_prefix}${aws_body}"

slack_prefix='xo''xb-'
slack_body='1234567890abcdef'
expect_fail_secret 'Slack token' "${slack_prefix}${slack_body}" "${slack_prefix}${slack_body}"

hf_prefix='h''f_'
hf_body='A1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6'
expect_fail_secret 'Hugging Face token' "${hf_prefix}${hf_body}" "${hf_prefix}${hf_body}"

# Banned names: the pattern and the match are made up and split in this file.
banned_pattern='zz''qx|wv''vk'
banned_word='wv''vk'
printf 'a line with %s in it\n' "$banned_word" > names.txt
git add names.txt
if HYGIENE_BANNED_PATTERN=$banned_pattern "$hygiene" > "$out" 2>&1; then
  fail 'banned name not detected'
fi
if grep -F -- "$banned_word" "$out" >/dev/null; then
  fail 'banned-name check printed the matched text'
fi
ok 'banned name detected'
git rm -q --cached names.txt
rm -f names.txt

if ! "$hygiene" > "$out" 2>&1; then
  cat "$out" >&2
  fail 'unset HYGIENE_BANNED_PATTERN should skip the name check'
fi
grep -q 'banned-name check skipped: HYGIENE_BANNED_PATTERN is not set' "$out" ||
  fail 'unset HYGIENE_BANNED_PATTERN did not print the skip notice'
ok 'banned-name check skipped when unset'

# A range whose commits add a credential file fails even though the tree is clean.
base=$(git rev-parse HEAD)
add_path old.key
git commit -q -m 'add key'
git rm -q -- old.key
git commit -q -m 'remove key'
expect_pass 'clean tree after the key was removed'
expect_fail 'commit range that added a credential path' "$base..HEAD"
expect_pass 'range with no added credential path' 'HEAD~1..HEAD'

printf 'all hygiene self-tests passed\n'
