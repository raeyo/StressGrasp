#!/usr/bin/env bash
# 학습 조건과 동일한 결정적 eval — hold 평균/방향별 (docs/EXP_RESIDUAL.md §8-3)
# usage: bash 43_hold_eval.sh <mode> <ckpt|-> <tag>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null
MODE="${1:?}"; CKPT="${2:?}"; [ "$CKPT" = "-" ] && CKPT=""; TAG="${3:?}"
OBJSET="${GS_OBJSET:-ours_bench/ours_S.yaml}"; NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
G=8; NENV=$((NOBJ*G)); OUT="$GS_ROOT/runs/residual/$TAG"; mkdir -p "$OUT"
cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" GS_ROOT="$GS_ROOT" \
GS_BRANCH=1 GS_BR_ALPHA=1 GS_BR_FRAME=palm GS_BR_REPLICAS=1 GS_BR_RESYNC=1 GS_BR_RESYNC_EVERY=5 \
GS_BR_SNAP="" GS_BR_DUMP="" GS_RES_BIN=1 GS_RES_DPLAN="${GS_RES_DPLAN:-0.5}" \
GS_RESIDUAL="$MODE" GS_RES_EVAL=1 GS_RES_CKPT="$CKPT" GS_RES_TAG="$TAG" GS_RES_OUT="$OUT" \
"$DG_PY" run_rl_grasp.py task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True \
  seed="${GS_SEED:-42}" task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" task.env.randomizeTrackingReference=True \
  task.env.randomizeGraspPose=True task.env.trackingReferenceFile="${GS_REF:-tasks/grasp_ref_inspire.pkl}" \
  task.env.trackingReferenceLiftTimestep=13 task.env.resetDofPosRandomInterval="${GS_DOFNOISE:-0}" \
  task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
  checkpoint="${GS_CKPT:-ckpt/inspire.pt}" > "$OUT/holdeval.log" 2>&1
grep -a "hold/dir\|eval round 9" "$OUT/holdeval.log" | tail -2
