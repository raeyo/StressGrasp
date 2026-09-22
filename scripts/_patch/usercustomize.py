"""인터프리터 시작 시 자동 로드 — graspstress 패치를 건다 (PYTHONPATH 에 이 디렉토리가 있을 때만)."""
try:
    import gs_stress
    gs_stress.install()
except Exception as _e:  # 패치 실패가 stock 실행을 막지 않게 한다
    import sys
    print("[gs] patch install failed: %r" % (_e,), file=sys.stderr)
