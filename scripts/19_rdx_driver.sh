#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?}"; CELLS="${2:?}"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while read -r obj mass g ph dose shift nenv seed; do
  [ -z "${obj:-}" ] && continue; case "$obj" in \#*) continue;; esac
  GS_GPU="$GPU" bash "$HERE/18_rdx.sh" "$obj" "$mass" "$g" "$ph" "$dose" "$shift" \
    "${nenv:-19}" "${seed:-42}" 2>&1 | tail -2
done < "$CELLS"
echo "[rdx gpu$GPU] ALL DONE"
