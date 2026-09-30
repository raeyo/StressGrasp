"""graspstress POSTRES 전용 패치 로더 — `_patch2` 를 PYTHONPATH 앞에 두면 이 파일이 `_patch/usercustomize.py` 를 가린다.
공유 하니스(`_patch/*`)는 한 줄도 고치지 않는다 (CLAUDE.md §2 — 소유 배정 없음)."""
import sys
for _m in ("gs_stress", "gs_branch", "gs_residual", "gs_postres"):
    try:
        __import__(_m).install()
    except Exception as _e:
        print("[gs] %s install failed: %r" % (_m, _e), file=sys.stderr)
