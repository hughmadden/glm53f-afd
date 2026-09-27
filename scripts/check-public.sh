#!/usr/bin/env bash
# Refuse to commit deployment-specific details to this public repository.
#
# Usage: scripts/check-public.sh [--staged]
#   PUBLIC_LEAK_PATTERNS  file of extended regular expressions, one per line
#                         (lines starting with # are comments). Required: the
#                         patterns describe a particular deployment, so they are
#                         kept outside this repository.
#   PUBLIC_LEAK_ALLOW     optional file of fixed strings; a matching line that
#                         contains one of them is allowed.
#
# With --staged, only the files staged for commit are checked (for a
# pre-commit hook); otherwise every tracked and untracked, non-ignored file is.
set -euo pipefail

patterns=${PUBLIC_LEAK_PATTERNS:-}
allow=${PUBLIC_LEAK_ALLOW:-}
if [[ -z "$patterns" || ! -r "$patterns" ]]; then
  echo "check-public: set PUBLIC_LEAK_PATTERNS to a readable pattern file" >&2
  exit 2
fi

root=$(git rev-parse --show-toplevel)
cd "$root"

if [[ "${1:-}" == "--staged" ]]; then
  mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR)
else
  mapfile -t files < <(git ls-files --cached --others --exclude-standard)
fi

regex=$(grep -vE '^[[:space:]]*(#|$)' "$patterns" | paste -sd '|' -)
[[ -n "$regex" ]] || { echo "check-public: no patterns in $patterns" >&2; exit 2; }

status=0
for f in "${files[@]}"; do
  [[ -f "$f" ]] || continue
  # file names
  if printf '%s\n' "$f" | grep -qiE "$regex"; then
    echo "check-public: forbidden pattern in file name: $f" >&2
    status=1
  fi
  # contents (text files only)
  grep -Iq . "$f" 2>/dev/null || continue
  while IFS= read -r hit; do
    if [[ -n "$allow" && -r "$allow" ]] && grep -qF -f <(grep -vE '^[[:space:]]*(#|$)' "$allow") <<<"$hit"; then
      continue
    fi
    echo "check-public: $f:$hit" >&2
    status=1
  done < <(grep -niE "$regex" "$f" || true)
done

if ((status)); then
  echo "check-public: FAILED (move deployment-specific notes to your private notes)" >&2
else
  echo "check-public: clean (${#files[@]} files)"
fi
exit "$status"
