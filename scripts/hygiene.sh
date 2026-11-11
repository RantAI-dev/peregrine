#!/usr/bin/env bash
# Repository hygiene: no credential files, no credential-shaped strings in tracked
# text files, and (when HYGIENE_BANNED_PATTERN is set) no banned names. With a
# commit range argument it also rejects a range in which any commit added a
# credential-style path, even when a later commit removed it.
#
# Usage: scripts/hygiene.sh [<base>..<head>]
#
# Matched secret text is never printed: findings are reported as path:line only.
# A line that is a deliberate false positive can carry a `hygiene:allow` comment.
set -eu

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "hygiene: not inside a git repository" >&2
  exit 2
fi
cd "$(git rev-parse --show-toplevel)"

findings=$(mktemp "${TMPDIR:-/tmp}/hygiene-findings.XXXXXX")
hits=$(mktemp "${TMPDIR:-/tmp}/hygiene-hits.XXXXXX")
trap 'rm -f "$findings" "$hits"' EXIT HUP INT TERM

# A path is credential-style when its basename is one of the known credential
# files. .env.example is allowed through here; the content scan below rejects it
# if it carries a non-placeholder value.
path_is_credential() {
  local base=${1##*/}
  case "$base" in
    .env) return 0 ;;
    .env.*)
      [ "$base" = .env.example ] && return 1
      return 0
      ;;
    *.pem|*.key|*.p12|*.pfx|*.kdbx|*.tfstate) return 0 ;;
    id_rsa*|id_ed25519*) return 0 ;;
    .netrc|.npmrc|.pypirc) return 0 ;;
    credentials*.json) return 0 ;;
  esac
  return 1
}

trim() {
  local s=$1
  s=${s#"${s%%[![:space:]]*}"}
  s=${s%"${s##*[![:space:]]}"}
  printf '%s' "$s"
}

# Decide whether an assignment's value is a real value rather than an empty,
# placeholder, numeric, boolean, or code expression. The line itself is never
# printed by any caller.
worth_reporting() {
  local content=$1 val lower
  val=${content#*=}
  val=$(trim "$val")
  case "$val" in
    \"*\") val=${val#\"}; val=${val%\"} ;;
    \'*\') val=${val#\'}; val=${val%\'} ;;
  esac
  val=$(trim "$val")
  val=${val%,}
  val=${val%;}
  val=$(trim "$val")
  [ -n "$val" ] || return 1
  lower=$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    null|none|nil|true|false|changeme|change-me|change_me|replace-me|replace_me|redacted|todo|placeholder|example|sample|dummy|not-needed|not_needed)
      return 1
      ;;
    your_*|your-*|'your '*) return 1 ;;
    '<'*'>'|'{{'*'}}') return 1 ;;
    '***'*|'...'*|'xxx'*) return 1 ;;
  esac
  # Code expressions and structured values, not a bare secret.
  case "$val" in
    *'['*|*']'*|*'('*|*')'*|*'{'*|*'}'*|*'`'*|*'$'*) return 1 ;;
  esac
  # Numbers and booleans are configuration values, not credentials.
  [[ "$val" =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]] && return 1
  # Member access such as args.token_tolerance.
  [[ "$val" =~ ^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$ ]] && return 1
  return 0
}

# 1. Credential-style paths, on the tree and on the commits in an optional range.
while IFS= read -r path; do
  [ -n "$path" ] || continue
  if path_is_credential "$path"; then
    printf '%s: credential-style file\n' "$path" >> "$findings"
  fi
done < <(git ls-files)

# 2. Credential-shaped strings in tracked text files.
# Private-key headers. Pieces keep this script from matching its own source.
pk_begin='-----BEGIN '
pk_end=' PRIVATE KEY-----'
key_re="${pk_begin}[A-Z0-9 ]*${pk_end}"

# Name=value where the name ends in one of the credential words, at any case,
# with underscore-separated prefixes or suffixes (SSH_PASS, DB_PASSWORD, ...).
assign_re='(^|[^A-Za-z0-9_])([A-Za-z0-9]+_)*(PASSWORD|PASS|SECRET|TOKEN|API_KEY)(_[A-Za-z0-9]+)*[[:space:]]*=[[:space:]]*[^[:space:]=]'

scan_re() {
  local label=$1 re=$2 hit path rest line content
  while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    path=${hit%%:*}
    rest=${hit#*:}
    line=${rest%%:*}
    content=${rest#*:}
    case "$content" in *hygiene:allow*) continue ;; esac
    printf '%s:%s: %s\n' "$path" "$line" "$label" >> "$findings"
  done < <(git grep -I -n -E -- "$re" -- . || true)
}

scan_assign() {
  local hit path rest line content
  while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    path=${hit%%:*}
    rest=${hit#*:}
    line=${rest%%:*}
    content=${rest#*:}
    case "$content" in *hygiene:allow*) continue ;; esac
    worth_reporting "$content" || continue
    printf '%s:%s: credential-shaped assignment\n' "$path" "$line" >> "$findings"
  done < <(git grep -I -n -i -E -- "$assign_re" -- . || true)
}

scan_re 'private-key header' "$key_re"
scan_assign
scan_re 'GitHub token' 'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'
scan_re 'API key token' 'sk-[A-Za-z0-9_-]{20,}'
scan_re 'AWS access key' 'AKIA[0-9A-Z]{16}'
scan_re 'Slack token' 'xox[baprs]-[A-Za-z0-9-]{10,}'
scan_re 'Hugging Face token' 'hf_[A-Za-z0-9]{30,}'

# 3. Banned names: the pattern lives only in the environment, never in the repo.
if [ -n "${HYGIENE_BANNED_PATTERN:-}" ]; then
  rc=0
  git grep -I -i -n -E -- "$HYGIENE_BANNED_PATTERN" -- . > "$hits" 2>/dev/null || rc=$?
  if [ "$rc" -gt 1 ]; then
    echo "hygiene: HYGIENE_BANNED_PATTERN is not a valid extended regular expression" >&2
    exit 2
  fi
  while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    path=${hit%%:*}
    rest=${hit#*:}
    line=${rest%%:*}
    printf '%s:%s: banned name\n' "$path" "$line" >> "$findings"
  done < "$hits"
else
  echo 'banned-name check skipped: HYGIENE_BANNED_PATTERN is not set'
fi

# 4. A range: no commit in it may add a credential-style path, even if a later
# commit removes the file again.
if [ "$#" -gt 0 ]; then
  if [ "$#" -ne 1 ]; then
    echo "usage: $0 [<base>..<head>]" >&2
    exit 2
  fi
  range=$1
  case "$range" in
    *..*) ;;
    *)
      echo "hygiene: range must be <base>..<head>" >&2
      exit 2
      ;;
  esac
  base=${range%%..*}
  head=${range##*..}
  if [ -n "$base" ] && ! git rev-parse --verify --quiet "$base^{commit}" >/dev/null; then
    echo "hygiene: unknown base commit: $base" >&2
    exit 2
  fi
  if [ -n "$head" ] && ! git rev-parse --verify --quiet "$head^{commit}" >/dev/null; then
    echo "hygiene: unknown head commit: $head" >&2
    exit 2
  fi
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    if path_is_credential "$path"; then
      printf '%s: credential-style file added in %s\n' "$path" "$range" >> "$findings"
    fi
  done < <(git log --diff-filter=A --name-only --format= "$range" --)
fi

if [ -s "$findings" ]; then
  echo "hygiene: failed" >&2
  sort -u "$findings" >&2
  echo "hygiene: remove the file, blank or replace the value, or add a 'hygiene:allow' comment on the line" >&2
  exit 1
fi

echo "hygiene: ok"
