# CLAIMS — 누가 무엇을 하고 있나

> 규칙 = [`CLAUDE.md`](CLAUDE.md). **이 파일은 hub(통합 세션)만 고친다.** 새 작업은 사용자를 통해 hub 에 배정을 요청한다.
> 공유 하니스(`scripts/_patch/*`, `_common.sh`)는 아래 "소유 파일" 열에 적힌 브랜치만 수정할 수 있다.

## 진행 중

| 브랜치 | 워크스페이스 | 주제 | 소유 파일 (수정 가능) | 스크립트 번호 | 상태 |
|---|---|---|---|---|---|
| `hub/setup` | hub (두 번째 세션) | 새 머신 재현 경로 (env/spec.sh·gates.sh · DemoGrasp 패치 · 첫 대상 lecun) | `env/*` · `third_party/*` · `SETUP.md` · `scripts/_common.sh` · `.gitignore` | — | **리뷰 대기** — 통합 세션 작업 트리에서 커밋돼 분리됨. `README.md` 변경은 정본 문서라 merge 시 통합 세션이 검토. ★ 이 세션은 **별도 worktree 로 옮긴 뒤** 이어서 작업 |

## 스크립트 번호 배정

| 범위 | 용도 |
|---|---|
| 10–43 | 기존 (Stage 0 · failmap · branch · residual) — 수정은 소유 배정 필요 |
| 35–39 | hub/postlift (완료, 35·36 사용) |
| 50–59 | kimm (배정 대기) |
| 60–69 | campus 서버 (배정 대기) |
| 70–79 | 예비 |

## 완료 (merge 됨)

| 브랜치 | 주제 | 결과 |
|---|---|---|
| `hub/postlift` | 결정 1 — 파지 후 다방향 안정성 | **B 결손 실재** ([`docs/RESULTS_POSTLIFT.md`](docs/RESULTS_POSTLIFT.md)) · `gs_branch.py` 소유 해제 |
