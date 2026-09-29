#!/usr/bin/env bash
# 파지 후(post-lift) 안정성 측정 — 설계 정본 docs/EXP_POSTLIFT.md · Stability_Metrics §2~§5
# usage: bash 35_posthold.sh <hand: inspire|shadow|allegro> <objset> <alpha> <tag> [seed]
#   alpha 하나 → G = 2 + 6 = 8 (control + 6방향 probe), num_envs = n_obj*8
#   tag 규약 <hand>_a<alpha>_s<seed> (36_posthold.py 가 파싱한다)
# env:
#   GS_TAU(기본 80)  : task.env.episodeLength. 창(t0+LOAD+RECOVER)이 끝나기 전에 에피소드가
#                      끝나지 않도록. 늘리는 것은 안전 (PROTOCOL.md §7.3).
#   GS_EXTRA         : 명령 끝에 붙는 추가 hydra 오버라이드 (공백 구분)
#   GS_BR_HOLD / GS_BR_LOAD / GS_BR_RAMP / GS_BR_RECOVER : py 기본 3/3/6/3 — env 로 그대로 전달됨
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; gs_check >/dev/null

HAND="${1:?}"; OBJSET="${2:?}"; ALPHA="${3:?}"; TAG="${4:?}"; SEED="${5:-42}"

# ★ 손별 인자는 원저자 규약(15_eval.sh) 그대로: inspire(=default)만 resetDofPosRandomInterval
#   를 주지 않고 lift 이후 timesteps 가 13, 나머지는 11 + resetDof=0.
case "$HAND" in
  inspire)  CKPT=ckpt/inspire.pt;     REF=tasks/grasp_ref_inspire.pkl; LIFT=13
            HARG=();        RDOF="" ;;
  shadow)   CKPT=ckpt/shadow.pt;      REF=tasks/grasp_ref_shadow.pkl;  LIFT=11
            HARG=(hand=shadow_simple); RDOF="task.env.resetDofPosRandomInterval=0" ;;
  allegro)  CKPT=ckpt/ur5_allegro.pt; REF=tasks/grasp_ref_allegro.pkl; LIFT=11
            HARG=(hand=ur5_allegro);   RDOF="task.env.resetDofPosRandomInterval=0" ;;
  *) echo "unknown hand: $HAND (inspire|shadow|allegro)" >&2; exit 1 ;;
esac

NOBJ="$(grep -c '^- ' "$DG_ROOT/assets/$OBJSET")"
G=$((2 + 6))                                   # control + 6방향 probe (alpha 1개)
NENV=$((NOBJ*G))

OUT="$GS_ROOT/runs/posthold/$TAG"; mkdir -p "$OUT"
echo "[ph] hand=$HAND n_obj=$NOBJ G=$G num_envs=$NENV alpha=$ALPHA tau=${GS_TAU:-80} gpu=$GS_GPU"

cd "$DG_ROOT"
CUDA_VISIBLE_DEVICES="$GS_GPU" PYTHONPATH="$GS_ROOT/scripts/_patch:${PYTHONPATH:-}" \
GS_BRANCH=1 GS_BR_OFFSET=posthold GS_BR_ALPHA="$ALPHA" GS_BR_FRAME=palm GS_BR_DUMP="$OUT/branch.npz" \
"$DG_PY" run_rl_grasp.py \
  task=grasp train=PPOOneStep test=True num_envs="$NENV" headless=True seed="$SEED" \
  task.env.observationType="eefpose+objinitpose+objpcl" task.env.armController=pose \
  task.env.asset.multiObjectList="$OBJSET" \
  task.env.randomizeTrackingReference=True task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile="$REF" \
  task.env.trackingReferenceLiftTimestep="$LIFT" ${HARG[@]+"${HARG[@]}"} $RDOF \
  task.env.episodeLength="${GS_TAU:-80}" task.env.enablePointCloud=True train.params.is_vision=True \
  ${GS_EXTRA:+$GS_EXTRA} checkpoint="$CKPT" > "$OUT/eval.log" 2>&1
RC=$?
echo "[ph] rc=$RC  log=$OUT/eval.log"
grep -a "^\[br\]" "$OUT/eval.log" | head -5
grep -ao "Round [0-9]*, success rate: [0-9.]*" "$OUT/eval.log" | tail -3
[ "$RC" != 0 ] && tail -25 "$OUT/eval.log"
exit 0
