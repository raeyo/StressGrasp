#!/usr/bin/env bash
# POSTRES — 폐합 후 상수 Δ 의 가동범위 측정. 설계 = docs/EXP_POSTRES.md.
# usage: GS_GPU=0 GS_HAND=shadow_simple bash scripts/postres_run.sh <tag> [episodes]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null
GS_PATCH2="$(cd "$(dirname "${BASH_SOURCE[0]}")/_patch2" && pwd)"

TAG="${1:?tag}"; EPS="${2:-20}"
OBJSET="${GS_OBJSET:-ours_bench/ours_S.yaml}"
K="${GS_PR_K:-3}"; REPL="${GS_REPL:-28}"; SEED="${GS_SEED:-42}"
ALPHA="${GS_ALPHA:-8}"; TAU="${GS_TAU:-70}"; REPS="${GS_PR_REPS:-2}"
HAND="${GS_HAND:-default}"
case "$HAND" in
  default)                  REF=tasks/grasp_ref_inspire.pkl; LIFT=13; HARG=(); CK=ckpt/inspire.pt; DOFN=0.2 ;;
  shadow_simple|fr3_shadow) REF=tasks/grasp_ref_shadow.pkl;  LIFT=11; HARG=(hand="$HAND"); CK=ckpt/shadow.pt; DOFN=0 ;;
  ur5_allegro)              REF=tasks/grasp_ref_allegro.pkl; LIFT=11; HARG=(hand="$HAND"); CK=ckpt/ur5_allegro.pt; DOFN=0 ;;
  *) echo "unknown hand: $HAND" >&2; exit 1 ;;
esac
DOFNOISE="${GS_DOFNOISE:-$DOFN}"          # posthold = 원저자 규약 (50_dataset.sh 와 동일)
NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
NA="$(awk -F, '{print NF}' <<< "$ALPHA")"; G=$((2 + 6*NA))
NMAIN=$((NOBJ*REPL)); NENV=$((NMAIN*G))
OUT="$GS_ROOT/runs/postres/$TAG"; mkdir -p "$OUT"
echo "[pr] tag=$TAG hand=$HAND n_obj=$NOBJ R=$REPL n_main=$NMAIN G=$G num_envs=$NENV eps=$EPS K=$K reps=$REPS seed=$SEED gpu=$GS_GPU cc=${GS_CC:-1} alpha=$ALPHA tau=$TAU dofnoise=$DOFNOISE"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_PATCH2:$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_ROOT="$GS_ROOT" \
GS_BRANCH=1 GS_BR_ALPHA="$ALPHA" GS_BR_FRAME=palm GS_BR_REPLICAS="$REPL" \
GS_BR_OFFSET=posthold GS_BR_RESYNC="" GS_BR_SNAP="" GS_BR_DUMP="" \
GS_POSTRES=1 GS_PR_MODE="${GS_PR_MODE:-sweep}" GS_PR_WORLD="${GS_PR_WORLD:-}" \
GS_PR_EPS="$EPS" GS_PR_K="$K" GS_PR_REPS="$REPS" GS_PR_SIGS="${GS_PR_SIGS:-0.05,0.10,0.20}" \
GS_PR_KCAP="${GS_PR_KCAP:-5}" GS_PR_LEVELS="${GS_PR_LEVELS:-0.01,0.05,0.15;0.02,0.10,0.25;0.03,0.20,0.40}" \
GS_PR_OUT="$OUT" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep "${HARG[@]}" test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="${GS_REF:-$REF}" \
  task.env.trackingReferenceLiftTimestep="${GS_LIFT:-$LIFT}" \
  task.env.resetDofPosRandomInterval="$DOFNOISE" \
  task.env.episodeLength="$TAU" task.env.enablePointCloud=True train.params.is_vision=True \
  task.sim.physx.contact_collection="${GS_CC:-1}" \
  checkpoint="${GS_CKPT:-$CK}" > "$OUT/pr.log" 2>&1
echo "[pr] rc=$?  log=$OUT/pr.log"
grep -a "^\[pr\]\|^\[br\]" "$OUT/pr.log" | head -5
tail -4 "$OUT/pr.log"
