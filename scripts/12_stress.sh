#!/usr/bin/env bash
# D2(질량)/D3(외력) 셀 — 런타임 패치 주입. 남의 코드는 고치지 않는다.
# usage: bash 12_stress.sh <ckpt> <objset> <tau> <seed> <nenv> <mass_scale> <ext_g>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
gs_check >/dev/null

CKPT="${1:?}"; OBJSET="${2:?}"; TAU="${3:?}"; SEED="${4:-42}"; NENV="${5:-66}"
MASS="${6:-1}"; EXTG="${7:-0}"

TAG="$(basename "$CKPT" .pt)__$(basename "$OBJSET" .yaml)__tau${TAU}__m${MASS}__g${EXTG}__seed${SEED}"
OUT="$GS_ROOT/runs/stress/$TAG"; mkdir -p "$OUT"
REF="${GS_REF:-tasks/grasp_ref_inspire.pkl}"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" \
PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_MASS_SCALE="$MASS" GS_EXT_G="$EXTG" GS_DUMP="$OUT/per_env.npz" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="$REF" task.env.trackingReferenceLiftTimestep=13 \
  task.env.episodeLength="$TAU" task.env.enablePointCloud=True train.params.is_vision=True \
  checkpoint="$CKPT" > "$OUT/eval.log" 2>&1

grep -o "Round [0-9]*, success rate: [0-9.]*" "$OUT/eval.log" > "$OUT/rounds.txt" || true
echo "[$TAG] rounds=$(wc -l < "$OUT/rounds.txt") mean=$(awk -F': ' '{s+=$2;n++} END{if(n)printf "%.4f", s/n; else print "NA"}' "$OUT/rounds.txt")"
