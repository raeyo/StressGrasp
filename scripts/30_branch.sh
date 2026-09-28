#!/usr/bin/env bash
# 분기 마진 M(s_t) 측정 — 설계 정본 docs/EXP_BRANCH.md
# usage: bash 30_branch.sh <ckpt> <objset> <alphas> <tag> [seed] [phase]
#   alphas="" 면 control 만 (G=2). "1" 이면 G=8. "1,2,5" 면 G=20.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null

CKPT="${1:?}"; OBJSET="${2:?}"; ALPHAS="${3-1}"; TAG="${4:?}"; SEED="${5:-42}"; PHASE="${6:-grip}"
NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
NA=0; [ -n "$ALPHAS" ] && NA="$(awk -F, '{print NF}' <<< "$ALPHAS")"
G=$((2 + 6*NA)); NENV=$((NOBJ*G))

OUT="$GS_ROOT/runs/branch/$TAG"; mkdir -p "$OUT"
echo "[br] n_obj=$NOBJ G=$G num_envs=$NENV alphas='$ALPHAS' phase=$PHASE gpu=$GS_GPU"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_BRANCH=1 GS_BR_ALPHA="$ALPHAS" GS_BR_PHASE="$PHASE" GS_BR_DUMP="$OUT/branch.npz" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="${GS_REF:-tasks/grasp_ref_inspire.pkl}" \
  task.env.trackingReferenceLiftTimestep="${GS_LIFT:-13}" ${GS_HAND:+hand=$GS_HAND} \
  task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
  checkpoint="$CKPT" > "$OUT/eval.log" 2>&1
RC=$?
echo "[br] rc=$RC  log=$OUT/eval.log"
grep -a "^\[br\]" "$OUT/eval.log" | head -5
grep -ao "Round [0-9]*, success rate: [0-9.]*" "$OUT/eval.log" | tail -3
[ "$RC" != 0 ] && tail -25 "$OUT/eval.log"
exit 0
