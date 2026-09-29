#!/usr/bin/env bash
# 버팀 카운트 라벨 생성 — 설계 docs/EXP_CRITIC.md §4.
# usage: bash 50_dataset.sh <tag> [episodes]   (GS_GPU / GS_REPL / GS_SEED 로 조절)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null

TAG="${1:?tag}"; EPS="${2:-40}"
OBJSET="${GS_OBJSET:-ours_bench/ours_S.yaml}"
REPL="${GS_REPL:-8}"; ALPHA="${GS_ALPHA:-1}"; H="${GS_H:-5}"; SEED="${GS_SEED:-42}"
NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
NA="$(awk -F, '{print NF}' <<< "$ALPHA")"; G=$((2 + 6*NA))
NMAIN=$((NOBJ*REPL)); NENV=$((NMAIN*G))
OUT="$GS_ROOT/runs/critic/$TAG"; mkdir -p "$OUT"
echo "[ds] tag=$TAG n_obj=$NOBJ R=$REPL n_main=$NMAIN G=$G num_envs=$NENV eps=$EPS seed=$SEED gpu=$GS_GPU"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_ROOT="$GS_ROOT" \
GS_BRANCH=1 GS_BR_ALPHA="$ALPHA" GS_BR_FRAME=palm GS_BR_REPLICAS="$REPL" \
GS_BR_RESYNC=1 GS_BR_RESYNC_EVERY="$H" GS_BR_SNAP="" GS_BR_DUMP="" \
GS_RESIDUAL=dataset GS_RES_BIN=1 GS_RES_THETA="${GS_THETA:-20}" \
GS_DS_EPS="$EPS" GS_DS_NPTS="${GS_NPTS:-256}" GS_DS_SIGS="${GS_SIGS:-0.05,0.10,0.20}" GS_DS_REPS="${GS_REPS:-3}" \
GS_RES_TAG="$TAG" GS_RES_OUT="$OUT" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="${GS_REF:-tasks/grasp_ref_inspire.pkl}" \
  task.env.trackingReferenceLiftTimestep="${GS_LIFT:-13}" \
  task.env.resetDofPosRandomInterval="${GS_DOFNOISE:-0.2}" \
  task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
  checkpoint="${GS_CKPT:-ckpt/inspire.pt}" > "$OUT/ds.log" 2>&1
echo "[ds] rc=$?  log=$OUT/ds.log"
grep -a "^\[res\]\|^\[br\]" "$OUT/ds.log" | head -3
tail -4 "$OUT/ds.log"
