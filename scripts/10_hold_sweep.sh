#!/usr/bin/env bash
# D1 유지시간 sweep — PROBLEM RQ1/RQ3, 반증 F1.
# 성공 판정식은 건드리지 않고 판정 시점(episodeLength)만 뒤로 민다. 코드 변경 0.
# usage: bash 10_hold_sweep.sh <ckpt> <objset> <tau> <seed> [num_envs]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
gs_check >/dev/null

CKPT="${1:?ckpt (예: ckpt/inspire.pt)}"
OBJSET="${2:?objset (예: ours_bench/ours_S.yaml)}"
TAU="${3:?tau (episodeLength steps)}"
SEED="${4:-0}"
NENV="${5:-66}"

TAG="$(basename "$CKPT" .pt)__$(basename "$OBJSET" .yaml)__tau${TAU}__seed${SEED}"
OUT="$GS_ROOT/runs/d1/$TAG"
mkdir -p "$OUT"

# 레퍼런스 파일은 embodiment 별로 다르다 (inspire 기본).
REF="${GS_REF:-tasks/grasp_ref_inspire.pkl}"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" "$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="$REF" task.env.trackingReferenceLiftTimestep=13 \
  task.env.episodeLength="$TAU" task.env.enablePointCloud=True train.params.is_vision=True \
  checkpoint="$CKPT" > "$OUT/eval.log" 2>&1

# 판독 근거만 추출 (터미널 덤프는 레포 밖에 남는다 — 우산 README §2)
grep -o "Round [0-9]*, success rate: [0-9.]*" "$OUT/eval.log" > "$OUT/rounds.txt" || true
echo "[$TAG] rounds=$(wc -l < "$OUT/rounds.txt")  mean=$(awk -F': ' '{s+=$2; n++} END{if(n)printf "%.4f", s/n; else print "NA"}' "$OUT/rounds.txt")"
