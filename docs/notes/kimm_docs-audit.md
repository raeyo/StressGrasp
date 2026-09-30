# 작업 노트 — 문서 감사·통합 (`kimm/docs`, 2026-09-30)

> 규칙상 작업 노트다 ([`CLAUDE.md`](../../CLAUDE.md) §2) — **판정·결론을 쓰지 않는다. 관측만.**
> 이 브랜치는 **제안**이다. 정본 문서(`PROBLEM*`·`PROTOCOL*`·`EXP_*`·`RESULTS_*`·`README`)의
> 확정·merge 는 **hub 소관**이다.

## 0. 무엇을 했나

문서 **23개 → 11개**(+ README). 축 = **"문제 정의를 다시 써도 살아남는가"**.
`Stability_Metrics.md` 는 **측정 정의 정본**이므로 손대지 않고 그대로 두었다.

| 남긴 것 | 역할 |
|---|---|
| `PROBLEM.md` | 문제 · D1~D5 · **결정 1(A/B/C)** · RQ · 반증기준 · v1 이력 |
| `Stability_Metrics.md` | ★ **측정 정의 정본** (지표 A~E · 프로토콜 · 선행 표 · 참고문헌) |
| `FACTS.md` | 우리가 잰 것 (진술·수치·**반증조건**) |
| `LITERATURE.md` | **선행 감시**만 (정의·서지는 `Stability_Metrics` 가 정본) |
| `PITFALLS.md` | 함정 P0~P20 · **null 체크리스트** |
| `ASSETS.md` | 데이터·모델·하네스·명령 · **없는 것** · 워크스페이스 배정 |
| `IDEAS.md` | 아이디어 대장 (**사망 원인 + 되살아나는 조건**) |
| `QUESTIONS.md` | 미결 |
| `METHOD.md` | 방법 · **§0 결정 1 과의 긴장** · 로드맵 |
| `PROTOCOL.md` | 측정 규약 · **§3-1 파지 후 프로토콜** · 판정 규율 · 재현 파라미터 |
| `EXPERIMENTS.md` | 실험 대장 E1~E11 + 무효 X1~X3 + 다음 T0~T6 |

## 1. 삭제한 파일과 흡수 위치 (전부 git history 에 남는다)

| 삭제 | 어디로 |
|---|---|
| `BASELINES.md` · `BASELINE_ACQUISITION.md` · `BASELINE_FITNESS.md` | `ASSETS.md` §2·§4 (4중 중복이었다) |
| `FAILURE_MAP.md` | `FACTS.md` A5·A6 + `PITFALLS.md` |
| `EXP_BRANCH.md` · `EXP_RESIDUAL.md` | 게이트·판정 → `EXPERIMENTS.md` · 재현 파라미터 → `PROTOCOL.md` §4·§7 |
| `RESULTS_STAGE0.md` · `RESULTS_BRANCH.md` · `RESULTS_RESIDUAL.md` | 헤드라인·기제 → `FACTS.md` · 판정 → `EXPERIMENTS.md`. **셀별 수치는 `results/*.md` 가 정본** |

★ **남긴 기준**: **살아 있는 사전등록–판정 쌍은 남긴다.** `EXP_POSTLIFT.md` + `RESULTS_POSTLIFT.md`
는 (a) master 에 merge 된 최신 판정이고 (b) [`CLAIMS.md`](../../CLAIMS.md) 가 링크하며
(c) `FACTS.md` A4 의 유일한 출처다. 나머지 쌍은 FACTS 로 흡수됐고, `RESULTS_BRANCH` 는 2손 재측정으로
곧 대체될 예정이다.

⚠ **hub 확인 요청**: 옛 `EXP_*`/`RESULTS_*` 삭제는 사전등록·판독 **원문**을 파일에서 없애는 것이다
(git history 에는 남는다). 정직성 장치로 파일을 유지해야 한다면 **이 브랜치의 해당 커밋만 되돌리면 된다.**

## 2. ⚠ 회수 불가 손실 1건

`docs/DISCUSSION_METHOD_2026-09-30.md`(20 KB, 09-30 02:30, `master` 의 공유 작업 디렉토리에서
**untracked** 상태였음)를 이 통합 과정에서 삭제했고 **git 에 없어 복구할 수 없다.**
내용은 흡수돼 있다: 사용자 결정 D1~D5 → `PROBLEM.md` §1 · "왜 기존 연구가 안 했나" 5묶음 →
`METHOD.md` §2 · ε 을 키우는 축 3개 → `METHOD.md` §4 · 두 라인 비교 → `METHOD.md` §5 ·
wrench 백본 비판 7건 → `IDEAS.md` §3 · 사다리 T1~T6 → `EXPERIMENTS.md` §3 · 안정성 7층 →
`Stability_Metrics.md` 가 더 정확한 판본으로 이미 갖고 있다.

## 3. 관측된 상충 (해석 없이 기록)

| # | 두 문서가 다르게 말한 것 | 대조로 확인된 쪽 |
|---|---|---|
| 1 | `BASELINES.md` §3 "RobustDexGrasp **포팅 미완**" vs `BASELINE_ACQUISITION.md` §2 "**포팅 완료**, 없는 것은 `RDX_ROOT`" | **후자.** `runs/_invalid_rdx_mass/` 가 실제 실행 기록이다 |
| 2 | `RESULTS_BRANCH.md` §3-3 Shadow lift SR **.476** vs `results/failmap.md` **.600** vs `RESULTS_POSTLIFT.md` **.609** | `30_branch.sh` 의 `resetDofPosRandomInterval=0` 누락 (**P16**) → 2손 마진 **인용 보류** |
| 3 | `RESULTS_BRANCH.md` §3-3 "순위 역전" vs `RESULTS_POSTLIFT.md` §1-3 "파지 후에는 재현되지 않는다" | 둘 다 사실. **구간이 다르다** — A2 는 형성 단계 한정 |
| 4 | `RESULTS_STAGE0.md` §4 "파지 후 낙폭 없음(`F1`)" vs `RESULTS_POSTLIFT.md` §0 "`B 결손 실재`" | 축이 다르다 — **무작위 수평 1방향 vs 6방향 전부** |
| 5 | `README.md`(master) "현재 단계 = Stage 0 문제 실재 판정" vs 결정 1 판정 완료 | README 가 뒤처져 있었다 |
| 6 | teacher 6방향 카운트가 **2.755 / 2.997 / 3.04 / 3.162 / 3.240** 다섯 값 | 조건이 다르다 → `FACTS.md` **B0** 에 조건과 함께 표로 |
| 7 | `EXP_BRANCH.md` §5 게이트 "M→성공 AUC ≥ .65" 통과 vs `RESULTS_BRANCH.md` §3-6 "순환적, 인용 금지" | **게이트 자체가 무의미**했다 → `FACTS.md` D4 |
| 8 | 파지 후 외력 knee 가 `RESULTS_STAGE0` **0.5530** vs `FAILURE_MAP` **0.5470** | 다른 셀. D4 에 따라 **N 단위로 통일** (15 N) |
| 9 | `BASELINE_FITNESS.md` §4 "주 baseline = `groot_cl_shadow_K8`" vs 이후 전 실험이 **teacher 단독** | 결정이 **미실행**. `ASSETS.md` §2-2 에 기록 |
| 10 | 함정이 `HANDOVER` §5(8건) · `FAILURE_MAP` §3(6건) · `METHOD` §5(6건) **세 벌, 서로 다른 번호** | `PITFALLS.md` 로 단일화 (P0~P20) |
| 11 | 시점 3분할 표가 `FAILURE_MAP` §1 과 `BASELINE_FITNESS` §5 에 **중복** | `FACTS.md` A6 하나로 |
| 12 | `METHOD.md`(`kimm/critic`) 는 **A(형성)** 를 겨냥 vs 결정 1 은 문제를 **B** 로 정의 | **미해소.** `METHOD.md` §0 에 셋(i/ii/iii)으로 열어 두었다 → **A→B 인과 측정이 선결** |

## 3-1. ⚠ 작업 중에 다른 세션이 움직였다 (2026-09-30 03:00~03:30)

공유 작업 디렉토리(`projects/graspstress`)에서 **다른 kimm 세션이 동시에 작업 중**이었다. 관측:
- 그 디렉토리의 브랜치가 `master` → **`kimm/dirpred`** 로 바뀌었다
- 커밋 `2d1decc`(문서 재편 반영)이 **이 통합 작업의 중간 산출물(신설 문서 7개)을 자기 커밋에 포함**시켰다
- **새 실험 DIRPRED**(7시간 자율, `docs/{EXP,RESULTS}_DIRPRED.md` · `scripts/52~54_*`)가 추가됐다
- `scripts/50_dataset.sh`·`_patch/gs_residual.py` 가 수정됐다 (K 지연 포착) — **하니스 소유 배정 없이**

→ **이 브랜치는 `origin/master` 기준이므로 DIRPRED 문서를 담지 않는다.**
E12 의 결과는 `FACTS.md` **§E**, `EXPERIMENTS.md` **E12**, `QUESTIONS.md` **Q6·Q9·Q12·Q13**,
`IDEAS.md` **I-15·I-32~I-34**, `PITFALLS.md` **P9·P21·P22**, `METHOD.md` §0 에 **브랜치를 명시해** 반영했다.

## 4. hub 에 확인을 요청하는 것

1. **정본 문서 4건**(`PROBLEM.md`·`PROTOCOL.md`·`README.md` 수정, `EXP_*`/`RESULTS_*` 삭제) — 확정 권한이 hub 다.
2. **상충 #12** — 방법을 A 로 유지할지 B/C 로 옮길지. `METHOD.md` §0 이 선택지만 정리했다.
3. **상충 #2** — 형성 단계 Shadow·Allegro **재측정 배정** (규약 지켜 3손, Q4·Q5 를 한 런으로).
4. **`kimm/critic` merge 여부** — `FACTS.md` §B·§C·§D 의 출처. 미merge 상태를 문서에 명시해 두었다.
5. ★ **`kimm/dirpred` merge 여부** — `FACTS.md` §E 의 출처이고, **하니스 수정이 배정 없이** 이뤄졌다.
   merge 순서에 따라 이 브랜치와 `docs/` 충돌이 생긴다 (그쪽 커밋 `2d1decc` 가 같은 신설 문서를 담고 있다).
   ★ **권장 순서: `kimm/critic` → `kimm/dirpred` → `kimm/docs`** (문서 통합을 마지막에).
6. **`CLAIMS.md` 갱신** — 이 브랜치(`kimm/docs`)를 "리뷰 요청"으로 등록. 그리고 **공유 디렉토리에
   두 kimm 세션이 동시에 있었던 것**을 §3-1 기준으로 정리.
7. ★ **E12 가 방법 선택에 주는 것** — 3고리 중 ②(예측)가 확정됐다. 후보 생성 라인의 안내자를
   스칼라 critic(ρ .457) → **6-dim 방향별 예측기**(AUC .75~.88)로 교체할 수 있다 (`IDEAS.md` I-32).
   ⚠ 단 **예측기가 손을 넘지 못한다**(교차 .55) — universal 주장에는 손 입력(I-33)이 선결.
