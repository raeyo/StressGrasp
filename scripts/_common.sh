# graspstress 공통 부팅 — 절대경로는 여기서만 흡수한다 (우산 README §3.6)
# 부팅 두 갈래: 우산 env.sh graspstress (새 머신, SETUP.md §3) 또는 이 파일만 source (kimm)
export GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VITAC_ROOT="${VITAC_ROOT:-$(cd "$GS_ROOT/../.." && pwd)}"

# ★ DG_ROOT 결정 순서: 이미 지정 → 새 머신 작업 사본(env/spec.sh 가 조립) → kimm 기존 사본.
#   external/DemoGrasp 원본(작업트리 오염)으로는 절대 폴백하지 않는다.
if [ -z "${DG_ROOT:-}" ] && [ -d "$GS_ROOT/third_party/DemoGrasp" ]; then
  export DG_ROOT="$GS_ROOT/third_party/DemoGrasp"
fi
export DG_ROOT="${DG_ROOT:-$VITAC_ROOT/projects/tacdexgrasp/references/DemoGrasp}"
# run_rl_grasp.py 훅(+debug=*)의 sys.path 재현 자산 — 패치가 이 변수 기반으로 참조한다
export DG_REPRO="${DG_REPRO:-$VITAC_ROOT/projects/tacdexgrasp/demograsp_repro}"
export DG_ENV="${DG_ENV:-$HOME/miniconda3/envs/demograsp}"
export DG_PY="$DG_ENV/bin/python"

# ★ GPU 정책 — kimm(공용, 0·1 만) 규칙의 일반화. env.local.sh 의 허용 목록을 먼저 읽는다
#   (변수 대입만 있는 로컬 파일 — 우산 env.sh 를 안 거치는 kimm 부팅용)
if [ -z "${VITAC_ALLOWED_GPUS:-}" ] && [ -f "$VITAC_ROOT/env.local.sh" ]; then
  . "$VITAC_ROOT/env.local.sh"
fi
export GS_GPU="${GS_GPU:-0}"

# isaacgym import 에 필요 (libpython3.8.so.1.0)
export LD_LIBRARY_PATH="$DG_ENV/lib:${LD_LIBRARY_PATH:-}"
export PATH="$DG_ENV/bin:$PATH"

gs_check() {
  [ -x "$DG_PY" ] || { echo "FATAL: python 없음: $DG_PY" >&2; return 1; }
  [ -d "$DG_ROOT" ] || { echo "FATAL: DemoGrasp 없음: $DG_ROOT — env/spec.sh 를 먼저 (SETUP.md §3)" >&2; return 1; }
  # 허용 목록 있으면 전부 그 안 · 머신 명시+비kimm 이면 자유 · 그 외(kimm·머신 불명)는 보수적으로 2·3 금지
  if [ -n "${VITAC_ALLOWED_GPUS:-}" ]; then
    for g in ${GS_GPU//,/ }; do
      case ",$VITAC_ALLOWED_GPUS," in
        *",$g,"*) ;;
        *) echo "FATAL: GPU $g 은(는) 허용 목록 밖 (VITAC_ALLOWED_GPUS=$VITAC_ALLOWED_GPUS)" >&2; return 1 ;;
      esac
    done
  elif [ -n "${VITAC_MACHINE:-}" ] && [ "$VITAC_MACHINE" != "kimm-h200" ]; then
    :  # 전용 머신 — 제한 없음
  else
    case ",${GS_GPU}," in *,2,*|*,3,*) echo "FATAL: GPU 2·3 은 타 사용자. 0·1 만." >&2; return 1;; esac
  fi
  echo "graspstress @ ${VITAC_MACHINE:-$(hostname -s)} · GPU=$GS_GPU · env=$DG_ENV"
}
