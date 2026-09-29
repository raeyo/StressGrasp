#!/usr/bin/env bash
# graspstress 환경 게이트 — env/spec.sh 와 같은 셸(우산 env.sh 부팅)에서 인자 없이 호출 (SETUP.md §3).
# 게이트별 한 줄 출력(G PASS/FAIL/SKIP). 하나라도 FAIL 이면 exit 1.
# G4 는 자산 부재 시 SKIP(18_rdx 만 못 씀), G5 는 GS_GATE_SIM=0 으로 SKIP.
set -uo pipefail
GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$GS_ROOT/scripts/_common.sh"   # DG_ROOT·DG_REPRO·GS_GPU·GPU 정책 (gs_check 은 G5 에서)

FAIL=0

# G1 — 인터프리터·torch 버전·CUDA
PV="$("$DG_PY" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
TV="$("$DG_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
CU="$("$DG_PY" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || true)"
G1=FAIL
case "$PV" in
  3.8.*) if [ "$TV" = "2.3.0+cu121" ] && [ "$CU" = "True" ]; then G1=PASS; fi ;;
esac
if [ "$G1" = PASS ]; then
  echo "G1 PASS python=$PV torch=$TV cuda=True"
else
  echo "G1 FAIL python='$PV' torch='$TV' cuda='$CU' — 기대: 3.8.x · 2.3.0+cu121 · True"
  FAIL=1
fi

# G2 — ★ isaacgym 은 torch 보다 먼저 import 해야 한다 (이 순서 그대로 검증)
if "$DG_PY" -c 'import isaacgym; from isaacgym import gymapi; import torch; import isaacgymenvs' >/dev/null 2>&1; then
  echo "G2 PASS import isaacgym→gymapi→torch→isaacgymenvs"
else
  echo "G2 FAIL import isaacgym→gymapi→torch→isaacgymenvs — env/spec.sh (c) 확인"
  FAIL=1
fi

# G3 — DG 작업 사본: 존재 · 패치 sha 일치 · ckpt/자산 해석(심볼릭 경유)
DGC="$GS_ROOT/third_party/DemoGrasp"
SHAWANT="$(sha256sum "$GS_ROOT/third_party/demograsp.patch" 2>/dev/null | cut -d' ' -f1)"
SHAGOT="$(cat "$DGC/.gs_patch_sha" 2>/dev/null || true)"
WHY=""
[ -d "$DGC/.git" ] || WHY="DG 사본 없음"
if ! { [ -n "$SHAWANT" ] && [ "$SHAWANT" = "$SHAGOT" ]; }; then WHY="${WHY:+$WHY · }패치 sha 불일치"; fi
[ -f "$DGC/ckpt/inspire.pt" ] || WHY="${WHY:+$WHY · }ckpt/inspire.pt 없음"
[ -f "$DGC/assets/ours_bench/ours_S.yaml" ] || WHY="${WHY:+$WHY · }assets/ours_bench/ours_S.yaml 없음"
if [ -z "$WHY" ]; then
  echo "G3 PASS DG 사본=$DGC (patch sha ${SHAWANT:0:12}…) · ckpt·자산 해석됨"
else
  echo "G3 FAIL $WHY — env/spec.sh (d) 재실행"
  FAIL=1
fi

# G4 — 18_rdx(RobustDexGrasp) 전용 자산. 없으면 SKIP (FAIL 아님)
RDX="${RDX_ROOT:-$VITAC_ROOT/external/RobustDexGrasp}"
if [ -f "$DG_REPRO/robustdex/eval_robustdex.py" ] && [ -f "$RDX/raisimGymTorch/data_all/student/student_ckpt/full_5500_r.pt" ]; then
  echo "G4 PASS robustdex 포팅($DG_REPRO/robustdex) · rdx ckpt(full_5500_r.pt)"
else
  echo "G4 SKIP 18_rdx 전용 자산 없음(robustdex 포팅 또는 RDX ckpt) — 그 셀만 못 씀"
fi

# G5 — sim 스모크: D1 한 셀을 실제로 돌려 rounds.txt 가 찍히는지 (GS_GPU 기본 0)
if [ "${GS_GATE_SIM:-1}" = "0" ]; then
  echo "G5 SKIP sim 스모크 (GS_GATE_SIM=0)"
else
  CKPT=ckpt/inspire.pt; OBJSET=ours_bench/ours_S.yaml; TAU=50; SEED=0; NENV=8
  TAG="$(basename "$CKPT" .pt)__$(basename "$OBJSET" .yaml)__tau${TAU}__seed${SEED}"
  OUT="$GS_ROOT/runs/d1/$TAG"
  if ! gs_check >/dev/null; then
    echo "G5 FAIL gs_check 실패 (위 FATAL 참고)"
    FAIL=1
  else
    # ★ 스모크 산출물이 runs/d1 집계(20_summarize.py)에 섞이지 않게 — 원래 없던 셀이면 판정 후 지운다
    PRE=0; [ -e "$OUT" ] && PRE=1
    bash "$GS_ROOT/scripts/10_hold_sweep.sh" "$CKPT" "$OBJSET" "$TAU" "$SEED" "$NENV" || true
    if [ -s "$OUT/rounds.txt" ]; then
      MEAN="$(awk -F': ' '{s+=$2;n++} END{if(n)printf "%.4f",s/n; else print "NA"}' "$OUT/rounds.txt")"
      echo "G5 PASS sim 스모크 $TAG · 평균 SR=$MEAN"
    else
      echo "G5 FAIL sim 스모크 $TAG — eval.log 마지막 20줄:"
      tail -n 20 "$OUT/eval.log" 2>/dev/null || echo "  (eval.log 없음: $OUT/eval.log)"
      FAIL=1
    fi
    [ "$PRE" = 1 ] || [ "$FAIL" = 1 ] || rm -rf "$OUT"   # 실패면 eval.log 를 남겨 둔다
  fi
fi

[ "$FAIL" = 0 ] || exit 1
echo "gates: 통과 (FAIL 없음)"
