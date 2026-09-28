#!/usr/bin/env bash
# 학습된 residual 을 **baseline 과 동일한 측정 프로토콜**로 평가한다 (docs/EXP_RESIDUAL.md §5).
# replicate-once · palm · α 사다리 · 6방향 · dose 1 env-step · 10 라운드.
# usage: bash 42_residual_eval.sh <mode> <ckpt|-> <tag>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null

MODE="${1:?min|sum|zero}"; CKPT="${2:?policy.pt 경로 또는 - }"; [ "$CKPT" = "-" ] && CKPT=""; TAG="${3:?}"
OBJSET="${GS_OBJSET:-ours_bench/ours_S.yaml}"
ALPHA="${GS_ALPHA:-0.125,0.25,0.5,1,2}"; SEED="${GS_SEED:-42}"
NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
NA="$(awk -F, '{print NF}' <<< "$ALPHA")"; G=$((2 + 6*NA)); NENV=$((NOBJ*G))
OUT="$GS_ROOT/runs/branch/$TAG"; mkdir -p "$OUT"
echo "[eval] mode=$MODE G=$G num_envs=$NENV ckpt=$CKPT gpu=$GS_GPU -> $OUT"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_ROOT="$GS_ROOT" \
GS_BRANCH=1 GS_BR_ALPHA="$ALPHA" GS_BR_FRAME=palm GS_BR_REPLICAS=1 \
GS_BR_SNAP="1,5,20" GS_BR_DUMP="$OUT/branch.npz" \
GS_RESIDUAL="$MODE" GS_RES_EVAL=1 GS_RES_CKPT="$CKPT" GS_RES_TAG="$TAG" GS_RES_OUT="$OUT" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="${GS_REF:-tasks/grasp_ref_inspire.pkl}" \
  task.env.trackingReferenceLiftTimestep="${GS_LIFT:-13}" \
  task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
  checkpoint="${GS_CKPT:-ckpt/inspire.pt}" > "$OUT/eval.log" 2>&1
echo "[eval] rc=$?  $(grep -ac 'eval round' "$OUT/eval.log") rounds"
grep -a "eval round" "$OUT/eval.log" | tail -3
