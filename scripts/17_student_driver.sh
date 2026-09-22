#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?}"; CELLS="${2:?}"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while read -r ck hand obj hl mass g ph dose shift seed nenv; do
  [ -z "${ck:-}" ] && continue; case "$ck" in \#*) continue;; esac
  GS_GPU="$GPU" bash "$HERE/14_student.sh" "$ck" "$hand" "$obj" "$hl" "$mass" "$g" "$ph" \
    "$dose" "$shift" "${seed:-42}" "${nenv:-66}" 2>&1 | tail -3
done < "$CELLS"
echo "[student gpu$GPU] ALL DONE"
