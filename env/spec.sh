#!/usr/bin/env bash
# graspstress 환경 구축 (새 머신) — 우산 env.sh graspstress 로 부팅한 셸에서 인자 없이 호출 (SETUP.md §3).
# 멱등: 각 단계는 이미 준비돼 있으면 skip. exit 0 = 구축 완료 (최종 판정은 env/gates.sh).
# 빌드 절차 원본 = projects/tacdexgrasp/demograsp_repro/SETUP.md §3 (torch cu121 · eigen · editable).
set -euo pipefail
GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 우산 변수 필수 — env.sh 미부팅이면 즉시 종료
: "${VITAC_EXTERNAL:?우산 env.sh graspstress 로 부팅 후 실행 (SETUP.md §3)}"
: "${VITAC_ENV:?우산 env.sh graspstress 로 부팅 후 실행 (SETUP.md §3)}"
: "${VITAC_CONDA:?우산 env.sh graspstress 로 부팅 후 실행 (SETUP.md §3)}"
: "${VITAC_SHARED:?우산 env.sh graspstress 로 부팅 후 실행 (SETUP.md §3)}"
: "${TMPDIR:?우산 env.sh graspstress 로 부팅 후 실행 (SETUP.md §3)}"
: "${PIP_CACHE_DIR:?우산 env.sh graspstress 로 부팅 후 실행 (SETUP.md §3)}"

# (a) 전제 — hub 가 rsync 로 심어둔 것들. 없으면 §2 부터.
MISSING=0
for p in \
  "$VITAC_EXTERNAL/isaacgym/python" \
  "$VITAC_EXTERNAL/IsaacGymEnvs" \
  "$VITAC_EXTERNAL/DemoGrasp/.git" \
  "$VITAC_SHARED/ckpt/demograsp/inspire.pt" \
  "$VITAC_SHARED/assets/ours_bench"; do
  [ -e "$p" ] || { echo "SPEC: 없음: $p"; MISSING=1; }
done
[ "$MISSING" = 0 ] || { echo "SPEC: hub 에서 sync 먼저 (SETUP.md §2)"; exit 1; }

# 이 레포 안의 재현 자료 (둘 다 커밋 대상 — 없으면 hub 에서 만들어 넣어야 한다)
PATCH="$GS_ROOT/third_party/demograsp.patch"
LOCK="$GS_ROOT/env/requirements.lock"
[ -f "$PATCH" ] || { echo "SPEC: 없음: $PATCH"; exit 1; }
[ -f "$LOCK" ] || { echo "SPEC: 없음: $LOCK"; exit 1; }

# (b) conda env — 새 머신은 Miniforge(conda-forge 기본). ★ conda activate 금지, 절대경로만.
PY="$VITAC_ENV/bin/python"
PIP="$VITAC_ENV/bin/pip"
if [ ! -x "$PY" ]; then
  [ -x "$VITAC_CONDA/bin/conda" ] || { echo "SPEC: conda 없음: $VITAC_CONDA/bin/conda"; exit 1; }
  "$VITAC_CONDA/bin/conda" create -y -p "$VITAC_ENV" -c conda-forge --override-channels python=3.8.19 eigen
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# (c) pip — torch(cu121) → lock → isaacgym·IsaacGymEnvs editable
TVOK=0
"$PY" - <<'EOF' >/dev/null 2>&1 && TVOK=1
import torch, torchvision, torchaudio
assert torch.__version__ == "2.3.0+cu121"
assert torchvision.__version__ == "0.18.0+cu121"
assert torchaudio.__version__ == "2.3.0+cu121"
EOF
if [ "$TVOK" = 0 ]; then
  "$PIP" install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
fi
# ★ CPLUS_INCLUDE_PATH — lock 안 sdist(pysdf 등)가 eigen 으로 컴파일된다
CPLUS_INCLUDE_PATH="$VITAC_ENV/include/eigen3" "$PIP" install -r "$LOCK"
"$PIP" show isaacgym >/dev/null 2>&1 || \
  CPLUS_INCLUDE_PATH="$VITAC_ENV/include/eigen3" "$PIP" install -e "$VITAC_EXTERNAL/isaacgym/python"
"$PIP" show isaacgymenvs >/dev/null 2>&1 || \
  CPLUS_INCLUDE_PATH="$VITAC_ENV/include/eigen3" "$PIP" install -e "$VITAC_EXTERNAL/IsaacGymEnvs"

# (d) DemoGrasp 작업 사본 — external 원본의 .git 에서 1255dc5 를 clone 하고 우리 패치를 얹는다.
#     패치 sha 가 바뀌면(갱신) 그 사본만 지우고 재조립. external 작업트리 자체는 건드리지 않는다.
DGC="$GS_ROOT/third_party/DemoGrasp"
SHAWANT="$(sha256sum "$PATCH" | cut -d' ' -f1)"
if [ -d "$DGC/.git" ] && [ "$(cat "$DGC/.gs_patch_sha" 2>/dev/null)" = "$SHAWANT" ]; then
  echo "SPEC: DG 사본 최신 (patch sha ${SHAWANT:0:12}…)"
else
  rm -rf "$DGC"
  git clone -q "$VITAC_EXTERNAL/DemoGrasp" "$DGC"
  git -C "$DGC" checkout -q 1255dc5
  git -C "$DGC" apply "$PATCH"
  rm -rf "$DGC/assets" "$DGC/ckpt"
  ln -s "$VITAC_SHARED/assets" "$DGC/assets"
  ln -s "$VITAC_SHARED/ckpt/demograsp" "$DGC/ckpt"
  echo "$SHAWANT" > "$DGC/.gs_patch_sha"
  echo "SPEC: DG 사본 조립 → $DGC"
fi

# (e) 한 줄 요약 — 전체 판정은 env/gates.sh
echo "SPEC: 완료 · python=$("$PY" -V 2>&1) · torch=$("$PY" -c 'import torch; print(torch.__version__)' 2>&1) · DG=$DGC"
