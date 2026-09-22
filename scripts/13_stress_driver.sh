#!/usr/bin/env bash
# stress 셀 목록 순차 실행. usage: bash 13_stress_driver.sh <gpu> <cells-file>
set -uo pipefail
GPU="${1:?}"; CELLS="${2:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while read -r ckpt objset tau seed nenv mass extg phase dose; do
  [ -z "${ckpt:-}" ] && continue
  case "$ckpt" in \#*) continue;; esac
  GS_GPU="$GPU" bash "$HERE/12_stress.sh" "$ckpt" "$objset" "$tau" "$seed" "$nenv" "$mass" "$extg" "${phase:-post}" "${dose:-0}" 2>&1 | tail -1
done < "$CELLS"
echo "[stress gpu$GPU] ALL DONE"
