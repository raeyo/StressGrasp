# CLAIMS — 누가 무엇을 하고 있나

> 규칙 = [`CLAUDE.md`](CLAUDE.md). **이 파일은 hub 만 고친다.** 새 작업은 사용자를 통해 hub 에 배정을 요청한다.
> 공유 하니스(`scripts/_patch/*`, `_common.sh`)는 아래 "소유 파일" 열에 적힌 브랜치만 수정할 수 있다.

## 진행 중

| 브랜치 | 워크스페이스 | 주제 | 소유 파일 (수정 가능) | 스크립트 번호 | 상태 |
|---|---|---|---|---|---|
| `hub/postlift` | hub | 결정 1 — 파지 후 다방향 안정성 측정 ([`docs/EXP_POSTLIFT.md`](docs/EXP_POSTLIFT.md)) | `scripts/_patch/gs_branch.py` · `scripts/35_posthold.sh` · `scripts/36_posthold.py` | 35–36 | 측정 중 (2026-09-29) |

## 스크립트 번호 배정

| 범위 | 용도 |
|---|---|
| 10–43 | 기존 (Stage 0 · failmap · branch · residual) — 수정은 소유 배정 필요 |
| 35–39 | hub/postlift |
| 50–59 | kimm (배정 대기) |
| 60–69 | campus 서버 (배정 대기) |
| 70–79 | 예비 |

## 완료 (merge 됨)

| 브랜치 | 주제 | merge 커밋 |
|---|---|---|
| — | — | — |
