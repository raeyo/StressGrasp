#!/usr/bin/env bash
# 범용 teacher 평가 — embodiment 7종 지원. 스트레스 인자는 선택.
# usage: bash 15_eval.sh <ckpt> <hand> <objset> <mass> <ext_g> <phase> <dose> <shift_cm> [seed] [nenv] [tau]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null
CKPT="${1:?}"; HAND="${2:?}"; OBJSET="${3:?}"; MASS="${4:-1}"; EXTG="${5:-0}"
PHASE="${6:-post}"; DOSE="${7:-0}"; SHIFT="${8:-0}"; SEED="${9:-42}"; NENV="${10:-66}"; TAU="${11:-50}"

EXTRA=()
case "$HAND" in
  # ★ 인자는 원저자 play_policy.sh 를 embodiment 별로 그대로 따른다.
  #   inspire 만 resetDofPosRandomInterval 을 지정하지 않는다 (원본이 그렇다).
  default)            REF=tasks/grasp_ref_inspire.pkl;        LIFT=13; HARG=() ;;
  shadow_simple|fr3_shadow) REF=tasks/grasp_ref_shadow.pkl;   LIFT=11; HARG=(hand="$HAND") ;;
  ur5_allegro)        REF=tasks/grasp_ref_allegro.pkl;        LIFT=11; HARG=(hand="$HAND") ;;
  ur5_svh)            REF=tasks/grasp_ref_svh.pkl;            LIFT=11; HARG=(hand="$HAND") ;;
  fr3_dclaw_gripper)  REF=tasks/grasp_ref_dclaw_gripper.pkl;  LIFT=11; HARG=(hand="$HAND") ;;
  fr3_panda_gripper)  REF=tasks/grasp_ref_panda_gripper.pkl;  LIFT=11; HARG=(hand="$HAND")
                      EXTRA+=('task.env.resetPositionRange=[[0.4,0.7],[-0.35,0.15],[0.1,0.12]]') ;;
  *) echo "unknown hand: $HAND" >&2; exit 1 ;;
esac
# panda_gripper 는 randomizeGraspPose 를 play_policy 에서 안 켠다
if [ "$HAND" = "fr3_panda_gripper" ]; then RGP=False; else RGP=True; fi
# inspire(default) 는 원본이 이 인자를 안 준다 — 주면 clean SR 이 0.7621 -> 0.7773 으로 바뀐다(실측)
if [ "$HAND" = "default" ]; then RESETDOF=""; else RESETDOF="task.env.resetDofPosRandomInterval=0"; fi

TAG="$(basename "$CKPT" .pt)__${HAND}__$(basename "$OBJSET" .yaml)__m${MASS}__g${EXTG}__${PHASE}d${DOSE}__s${SHIFT}__seed${SEED}"
OUT="$GS_ROOT/runs/grid/$TAG"; mkdir -p "$OUT"
cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_MASS_SCALE="$MASS" GS_EXT_G="$EXTG" GS_PHASE="$PHASE" GS_FORCE_STEPS="$DOSE" \
GS_OBJ_SHIFT="$SHIFT" GS_DUMP="$OUT/per_env.npz" \
"$DG_PY" run_rl_grasp.py task=grasp train=PPOOneStep "${HARG[@]}" test=True \
  num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=$RGP \
  $RESETDOF \
  task.env.trackingReferenceFile="$REF" task.env.trackingReferenceLiftTimestep="$LIFT" \
  task.env.episodeLength="$TAU" task.env.enablePointCloud=True train.params.is_vision=True \
  "${EXTRA[@]}" checkpoint="$CKPT" > "$OUT/eval.log" 2>&1 || true
grep -o "Round [0-9]*, success rate: [0-9.]*" "$OUT/eval.log" > "$OUT/rounds.txt" || true
echo "[$TAG] rounds=$(wc -l < "$OUT/rounds.txt") mean=$(awk -F': ' '{s+=$2;n++} END{if(n)printf "%.4f",s/n; else print "NA"}' "$OUT/rounds.txt")"
