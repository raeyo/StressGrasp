#!/usr/bin/env bash
# D1 sweep 드라이버 — 한 GPU 에서 셀 목록을 순차 실행. usage: bash 11_d1_driver.sh <gpu> <cells-file>
set -uo pipefail
GPU="${1:?gpu}"; CELLS="${2:?cells file}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while read -r ckpt objset tau seed nenv; do
  [ -z "${ckpt:-}" ] && continue
  case "$ckpt" in \#*) continue;; esac
  GS_GPU="$GPU" bash "$HERE/10_hold_sweep.sh" "$ckpt" "$objset" "$tau" "$seed" "$nenv" 2>&1 | tail -1
done < "$CELLS"
echo "[driver gpu$GPU] ALL DONE"
