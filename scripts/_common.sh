# graspstress 공통 부팅 — 절대경로는 여기서만 흡수한다 (우산 README §3.6)
export GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VITAC_ROOT="${VITAC_ROOT:-$(cd "$GS_ROOT/../.." && pwd)}"

# 평가 대상 env = DemoGrasp (남의 코드. 고치지 않는다)
export DG_ROOT="${DG_ROOT:-$VITAC_ROOT/projects/tacdexgrasp/references/DemoGrasp}"
export DG_ENV="${DG_ENV:-$HOME/miniconda3/envs/demograsp}"
export DG_PY="$DG_ENV/bin/python"

# ★ 공용 서버 — GPU 는 0·1 만 (env.local.sh VITAC_ALLOWED_GPUS). 2·3 은 다른 사용자.
export GS_GPU="${GS_GPU:-0}"

# isaacgym import 에 필요 (libpython3.8.so.1.0)
export LD_LIBRARY_PATH="$DG_ENV/lib:${LD_LIBRARY_PATH:-}"
export PATH="$DG_ENV/bin:$PATH"

gs_check() {
  [ -x "$DG_PY" ] || { echo "FATAL: python 없음: $DG_PY" >&2; return 1; }
  [ -d "$DG_ROOT" ] || { echo "FATAL: DemoGrasp 없음: $DG_ROOT" >&2; return 1; }
  case ",${GS_GPU}," in *,2,*|*,3,*) echo "FATAL: GPU 2·3 은 타 사용자. 0·1 만." >&2; return 1;; esac
  echo "graspstress @ kimm-h200 · GPU=$GS_GPU · env=$DG_ENV"
}
