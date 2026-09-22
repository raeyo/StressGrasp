#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?}"; CELLS="${2:?}"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while read -r ck hand obj mass g ph dose shift seed nenv tau; do
  [ -z "${ck:-}" ] && continue; case "$ck" in \#*) continue;; esac
  GS_GPU="$GPU" bash "$HERE/15_eval.sh" "$ck" "$hand" "$obj" "$mass" "$g" "$ph" "$dose" "$shift" \
    "${seed:-42}" "${nenv:-66}" "${tau:-50}" 2>&1 | tail -1
done < "$CELLS"
echo "[grid gpu$GPU] ALL DONE"
