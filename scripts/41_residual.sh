#!/usr/bin/env bash
# residual RL — 설계 docs/EXP_RESIDUAL.md.  usage: bash 41_residual.sh <mode> <tag> [iters] [extra env]
#   mode = min | sum | zero(대조군: Δ=0 으로 고정, 같은 루프를 돌려 하네스 효과를 분리)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null

MODE="${1:?min|sum|zero}"; TAG="${2:?}"; ITERS="${3:-150}"
OBJSET="${GS_OBJSET:-ours_bench/ours_S.yaml}"
REPL="${GS_REPL:-4}"; ALPHA="${GS_ALPHA:-1}"; H="${GS_H:-5}"; SEED="${GS_SEED:-42}"
NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
NA="$(awk -F, '{print NF}' <<< "$ALPHA")"; G=$((2 + 6*NA))
NMAIN=$((NOBJ*REPL)); NENV=$((NMAIN*G))
OUT="$GS_ROOT/runs/residual/$TAG"; mkdir -p "$OUT"
echo "[res] mode=$MODE n_obj=$NOBJ repl=$REPL n_main=$NMAIN G=$G num_envs=$NENV H=$H alpha=$ALPHA iters=$ITERS gpu=$GS_GPU"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_ROOT="$GS_ROOT" \
GS_BRANCH=1 GS_BR_ALPHA="$ALPHA" GS_BR_FRAME=palm GS_BR_REPLICAS="$REPL" \
GS_BR_RESYNC=1 GS_BR_RESYNC_EVERY="$H" GS_BR_SNAP="" GS_BR_DUMP="" \
GS_RESIDUAL="$MODE" GS_RES_ITERS="$ITERS" GS_RES_TAG="$TAG" GS_RES_OUT="$OUT" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="${GS_REF:-tasks/grasp_ref_inspire.pkl}" \
  task.env.trackingReferenceLiftTimestep="${GS_LIFT:-13}" \
  task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
  checkpoint="${GS_CKPT:-ckpt/inspire.pt}" > "$OUT/train.log" 2>&1
RC=$?
echo "[res] rc=$RC  log=$OUT/train.log"
grep -a "^\[res\]\|^\[br\]" "$OUT/train.log" | head -4
tail -5 "$OUT/train.log"
exit 0
