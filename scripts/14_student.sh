#!/usr/bin/env bash
# 주 baseline — GR00T student. ★ 같은 ckpt 를 exec_H 로 open/closed-loop 둘 다 평가한다
#   exec_H=0  → OPEN-LOOP  (frame-0 관측 → K-step 계획 전부 재생)
#   exec_H=H  → CLOSED-LOOP(H 스텝마다 재관측 → 새 chunk → 앞 H 실행)
# 가중치·구조·데이터가 전부 같고 **반응성만** 다르다 (docs/BASELINE_FITNESS.md §3).
# usage: bash 14_student.sh <ckpt_path> <hand> <objset> <H_list> <mass> <ext_g> <phase> [seed] [nenv]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
gs_check >/dev/null

SCKPT="${1:?student ckpt 절대경로}"; HAND="${2:?hand (shadow_simple|fr3_inspire_tac|...)}"
OBJSET="${3:?}"; HLIST="${4:-0}"; MASS="${5:-1}"; EXTG="${6:-0}"; PHASE="${7:-post}"
DOSE="${8:-0}"; SHIFT="${9:-0}"; SEED="${10:-42}"; NENV="${11:-66}"

case "$HAND" in
  shadow_simple|fr3_shadow) REF=tasks/grasp_ref_shadow.pkl; LIFT=11 ;;
  ur5_allegro)              REF=tasks/grasp_ref_allegro.pkl; LIFT=11 ;;
  *)                        REF=tasks/grasp_ref_inspire.pkl; LIFT=13 ;;
esac

TAG="$(basename "$(dirname "$SCKPT")")__${HAND}__$(basename "$OBJSET" .yaml)__H${HLIST//,/-}__m${MASS}__g${EXTG}__${PHASE}d${DOSE}__s${SHIFT}__seed${SEED}"
OUT="$GS_ROOT/runs/student/$TAG"; mkdir -p "$OUT"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" \
PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_MASS_SCALE="$MASS" GS_EXT_G="$EXTG" GS_PHASE="$PHASE" GS_FORCE_STEPS="$DOSE" \
GS_OBJ_SHIFT="$SHIFT" GS_DUMP="$OUT/per_env.npz" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep hand="$HAND" test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.resetDofPosRandomInterval=0 \
  task.env.trackingReferenceFile="$REF" task.env.trackingReferenceLiftTimestep="$LIFT" \
  task.env.episodeLength=50 task.env.enablePointCloud=True train.params.is_vision=True \
  +debug=eval_groot +student_ckpt="$SCKPT" +exec_H="$HLIST" \
  > "$OUT/eval.log" 2>&1

grep -a "=== .* SR = " "$OUT/eval.log" | tee "$OUT/sr.txt"
echo "[$TAG] done"
