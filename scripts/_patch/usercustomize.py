"""인터프리터 시작 시 자동 로드 — graspstress 패치를 건다 (PYTHONPATH 에 이 디렉토리가 있을 때만)."""
import sys
for _m in ("gs_stress", "gs_branch", "gs_residual"):
    try:
        __import__(_m).install()
    except Exception as _e:  # 패치 실패가 stock 실행을 막지 않게 한다
        print("[gs] %s install failed: %r" % (_m, _e), file=sys.stderr)
