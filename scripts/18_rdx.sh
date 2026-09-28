#!/usr/bin/env bash
# RobustDexGrasp (closed-loop 5Hz) — IsaacGym 포팅본. 원본 레포 = external/RobustDexGrasp (실물)
# ★ sim2sim gap 주의: obs 에 접촉 임펄스·관절 토크가 들어가고 정규화 통계는 RaiSim 롤아웃 기반이다.
#   낮은 SR 이 "env 가 쉽다"의 반증이 아니라 전이 gap 일 수 있다 (ROBUSTDEXGRASP_PORT.md).
#   → 원본 RaiSim SR 0.969 (raisim_baseline/) 와 대조해 포팅 건전성부터 본다.
# usage: bash 18_rdx.sh <objset> <mass> <ext_g> <phase> <dose> <shift_cm> [nenv] [seed]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null
export RDX_ROOT="${RDX_ROOT:-$VITAC_ROOT/external/RobustDexGrasp}"
OBJSET="${1:?}"; MASS="${2:-1}"; EXTG="${3:-0}"; PHASE="${4:-post}"; DOSE="${5:-0}"
SHIFT="${6:-0}"; NENV="${7:-19}"; SEED="${8:-42}"

TAG="rdx__ur5_allegro__$(basename "$OBJSET" .yaml)__m${MASS}__g${EXTG}__${PHASE}d${DOSE}__s${SHIFT}__seed${SEED}"
OUT="$GS_ROOT/runs/rdx/$TAG"; mkdir -p "$OUT"
cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
RDX_ROOT="$RDX_ROOT" \
GS_MASS_SCALE="$MASS" GS_EXT_G="$EXTG" GS_PHASE="$PHASE" GS_FORCE_STEPS="$DOSE" \
GS_OBJ_SHIFT="$SHIFT" GS_DUMP="$OUT/per_env.npz" \
"$DG_PY" run_rl_grasp.py task=grasp train=PPOOneStep hand=ur5_allegro test=True \
  num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile=tasks/grasp_ref_allegro.pkl \
  task.env.trackingReferenceLiftTimestep=11 task.env.resetDofPosRandomInterval=0 \
  task.env.episodeLength=50 task.env.enablePointCloud=True \
  +debug=eval_robustdex +rdx_updates=8 +rdx_grasp_steps=70 +rdx_lift_steps=100 \
  > "$OUT/eval.log" 2>&1
tail -20 "$OUT/eval.log" | grep -ai "SR\|success" | tail -5
echo "[$TAG] log=$OUT/eval.log"
