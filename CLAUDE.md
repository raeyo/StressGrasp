# graspstress — 작업 규칙 (모든 워크스페이스·모든 세션 공통)

이 레포(`raeyo/StressGrasp`)는 **여러 워크스페이스에서 동시에** 작업한다 (hub 로컬 · kimm-h200 · campus 서버).
작업이 겹치지 않도록 아래 git 규칙을 **반드시** 지킨다. 현재 누가 무엇을 하는지는 [`CLAIMS.md`](CLAIMS.md).

## 0. 역할

| 워크스페이스 | 태그 | 역할 |
|---|---|---|
| hub (로컬 raeyo-pc, RTX 3080) | `hub` | **통합 담당.** master merge · 정본 문서 · 판정 · CLAIMS 배정 |
| kimm-h200 (git 전용, GPU 0·1) | `kimm` | 배정받은 실험 실행 · 코드 |
| campus 서버 (bengio 등) | `bengio` 등 머신 alias | 배정받은 실험 실행 · 코드 |

## 1. 브랜치

- ★ **작업 디렉토리 하나 = 세션 하나 = 브랜치 하나.** 같은 머신에서 세션을 둘 이상 돌리면 두 번째부터는
  반드시 별도 worktree 를 쓴다: `git worktree add ../graspstress-<주제> -b <태그>/<주제> origin/master`.
  남의 세션이 체크아웃해 둔 디렉토리에서 커밋하면 **그 세션의 브랜치에 커밋이 섞인다** (2026-09-29 실제 발생:
  setup 커밋이 `hub/postlift` 위에 올라감 → hub 가 분리함). 커밋 전 `git branch --show-current` 가
  **자기 브랜치인지** 확인한다.
- **master 에는 hub 만 push 한다.** 다른 워크스페이스는 master 에 커밋·push 하지 않는다.
- 작업은 `<태그>/<주제>` 브랜치에서만 한다 (예: `kimm/rdx-port`, `hub/postlift`). **CLAIMS.md 에 배정된 브랜치만** 만든다.
- 시작: `git fetch origin && git switch -c <태그>/<주제> origin/master`
- 동기화: 자기 브랜치에서 `git fetch origin && git rebase origin/master`. `--force-with-lease` 는 **자기 브랜치에만** 허용.
- 남의 브랜치에 커밋하지 않는다. 필요한 것이 있으면 hub 에 요청한다 (사용자를 통해).
- 끝나면 브랜치를 push 하고 CLAIMS.md 의 해당 행을 "리뷰 요청"으로 바꿔 달라고 사용자에게 알린다. merge 는 hub 가 검증 후 한다.

## 2. 파일 소유권 — 겹침 방지의 핵심

| 범주 | 경로 | 누가 고치나 |
|---|---|---|
| 정본 문서 | `docs/PROBLEM*.md` · `docs/PROTOCOL*.md` · `docs/EXP_*.md`(판정 규칙) · `docs/RESULTS_*.md` · `README.md` · `CLAUDE.md` · `CLAIMS.md` | **hub 만** |
| 공유 하니스 | `scripts/_patch/*.py` · `scripts/_common.sh` | **CLAIMS.md 에서 그 파일을 배정받은 한 브랜치만.** 배정 없으면 수정 금지 — 새 패치 파일을 만들거나 hub 에 요청 |
| 새 스크립트 | `scripts/NN_*.{sh,py}` | 번호 NN 은 **CLAIMS.md 에서 배정받은 범위**만 쓴다 (번호 충돌 방지) |
| 작업 노트 | `docs/notes/<태그>_<주제>.md` | 각자 자유 (판정·결론은 쓰지 않는다 — 관측만) |
| 결과 요약 | `results/<실험>__<태그>.md/.json` | 각자. **파일명에 반드시 워크스페이스 태그** — 머신 간 절대값 비교 금지 (PROTOCOL §6) |
| 원시 로그 | `runs/` | 커밋 금지 (gitignore) |

★ 판정 규칙(임계값)은 **측정 전에 hub 가 EXP 문서로 master 에 커밋**한다. 다른 워크스페이스는 그 커밋 이후에 측정을 시작하고, 결과 파일에 기준 커밋 해시를 적는다.

## 3. 커밋

- 메시지 첫머리에 워크스페이스 태그: `[kimm] RDX 포팅 SR 0.969 재현` · `[hub] posthold 모드 추가`
- 한 커밋 = 한 논리 단위. 코드 변경과 결과 요약은 가능하면 분리.
- 커밋 전 `git status` 로 **소유권 밖 파일이 섞였는지** 확인한다. 섞였으면 빼고 커밋한다.

## 4. 환경 주의

- DemoGrasp 는 남의 코드다 — 고치지 않는다. **로컬 `external/DemoGrasp` 는 수정본**(`plan_anchor_pos` 등 추가)이고
  kimm 은 원본 `tacdexgrasp/references/DemoGrasp` 다. 하니스는 두 쪽 모두에서 돌아야 한다 (없는 속성은 getattr 로 건너뛴다).
- kimm-h200: GPU **0·1 만**. ssh 불가 → 모든 교환은 GitHub 로.
