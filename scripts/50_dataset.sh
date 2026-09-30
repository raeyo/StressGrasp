#!/usr/bin/env bash
# 버팀 카운트 라벨 생성 — 설계 docs/EXP_CRITIC.md §4.
# usage: bash 50_dataset.sh <tag> [episodes]   (GS_GPU / GS_REPL / GS_SEED 로 조절)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null

TAG="${1:?tag}"; EPS="${2:-40}"
OBJSET="${GS_OBJSET:-ours_bench/ours_S.yaml}"
REPL="${GS_REPL:-8}"; H="${GS_H:-5}"; SEED="${GS_SEED:-42}"
# ★ GS_PH=1: 파지 후(posthold) 라벨 모드 (EXP_HOLDPRED.md). α 기본 8, 에피소드 70, 반복 2, resync 없음.
PH="${GS_PH:-0}"
if [ "$PH" = "1" ]; then ALPHA="${GS_ALPHA:-8}"; TAU="${GS_TAU:-70}"; REPS="${GS_REPS:-2}"
  BR_OFFSET=posthold; BR_RESYNC=""
else ALPHA="${GS_ALPHA:-1}"; TAU=50; REPS="${GS_REPS:-3}"
  BR_OFFSET=auto; BR_RESYNC=1; fi
# ★ 손 선택 (GS_HAND). 인자는 15_eval.sh 의 손별 recipe 를 그대로 따른다.
HAND="${GS_HAND:-default}"
case "$HAND" in
  default)                  REF=tasks/grasp_ref_inspire.pkl; LIFT=13; HARG=(); CK=ckpt/inspire.pt; DOFN=0.2 ;;
  shadow_simple|fr3_shadow) REF=tasks/grasp_ref_shadow.pkl;  LIFT=11; HARG=(hand="$HAND"); CK=ckpt/shadow.pt; DOFN=0 ;;
  ur5_allegro)              REF=tasks/grasp_ref_allegro.pkl; LIFT=11; HARG=(hand="$HAND"); CK=ckpt/ur5_allegro.pt; DOFN=0 ;;
  *) echo "unknown hand: $HAND" >&2; exit 1 ;;
esac
# reset DOF 잡음: PH 모드는 원저자 규약(15_eval.sh / hub 35_posthold.sh)을 따른다 — Inspire 0.2(yaml 기본), 나머지 0.
# 형성 구간 모드(DIRPRED 이전)는 종전대로 모든 손 0.2.
if [ "$PH" = "1" ]; then DOFNOISE="${GS_DOFNOISE:-$DOFN}"; else DOFNOISE="${GS_DOFNOISE:-0.2}"; fi
NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
NA="$(awk -F, '{print NF}' <<< "$ALPHA")"; G=$((2 + 6*NA))
NMAIN=$((NOBJ*REPL)); NENV=$((NMAIN*G))
OUT="$GS_ROOT/runs/critic/$TAG"; mkdir -p "$OUT"
echo "[ds] tag=$TAG hand=$HAND n_obj=$NOBJ R=$REPL n_main=$NMAIN G=$G num_envs=$NENV eps=$EPS seed=$SEED gpu=$GS_GPU cc=${GS_CC:-0} kcap=${GS_DS_CAPTURE_K:-5} ph=$PH alpha=$ALPHA tau=$TAU reps=$REPS dofnoise=$DOFNOISE cands=${GS_DS_CANDS:-}"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_ROOT="$GS_ROOT" \
GS_BRANCH=1 GS_BR_ALPHA="$ALPHA" GS_BR_FRAME=palm GS_BR_REPLICAS="$REPL" \
GS_BR_OFFSET="$BR_OFFSET" GS_BR_RESYNC="$BR_RESYNC" GS_BR_RESYNC_EVERY="$H" GS_BR_SNAP="" GS_BR_DUMP="" \
GS_RESIDUAL=dataset GS_RES_BIN=1 GS_RES_THETA="${GS_THETA:-20}" \
GS_DS_EPS="$EPS" GS_DS_NPTS="${GS_NPTS:-256}" GS_DS_SIGS="${GS_SIGS:-0.05,0.10,0.20}" GS_DS_REPS="$REPS" \
GS_RES_TAG="$TAG" GS_RES_OUT="$OUT" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep "${HARG[@]}" test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="${GS_REF:-$REF}" \
  task.env.trackingReferenceLiftTimestep="${GS_LIFT:-$LIFT}" \
  task.env.resetDofPosRandomInterval="$DOFNOISE" \
  task.env.episodeLength="$TAU" task.env.enablePointCloud=True train.params.is_vision=True \
  task.sim.physx.contact_collection="${GS_CC:-0}" \
  checkpoint="${GS_CKPT:-$CK}" > "$OUT/ds.log" 2>&1
echo "[ds] rc=$?  log=$OUT/ds.log"
grep -a "^\[res\]\|^\[br\]" "$OUT/ds.log" | head -3
tail -4 "$OUT/ds.log"
