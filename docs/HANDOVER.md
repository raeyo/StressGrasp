# graspstress 세션 인계 (2026-09-30 작성)

> **여기서부터 읽어라.** 문서 위계:
> 이 파일 → [`METHOD.md`](METHOD.md)(★방법 정본·현행) → [`PROBLEM.md`](PROBLEM.md)(문제 정의, 불변)
> → `EXP_*.md`(사전등록) → `RESULTS_*.md`(판독) → `../results/`(수치)
>
> 레포 = `git@github.com:raeyo/StressGrasp.git`. ★ **디렉토리명 `graspstress` ≠ 레포명 `StressGrasp`** —
> 새 머신에서 clone 하면 이름을 `graspstress` 로 맞춰야 스크립트 경로가 맞는다.

## 0. 부팅

```bash
cd $VITAC_ROOT/projects/graspstress && source scripts/_common.sh && gs_check
# python = $DG_PY ($HOME/miniconda3/envs/demograsp, py3.8). GPU 는 0·1 만 (공용 서버, 2·3 은 타 사용자)
# 평가 env = references/DemoGrasp (IsaacGym Preview 4) — ★ 남의 코드, 고치지 않는다. 런타임 패치로만 얹는다
```

## 1. 한 문단 위치

**Stage 0(문제 실재 판정)은 끝났다** — 파지는 한 옥탄트만 버티고(60~80배 이방성), lift SR 은 버팀을
예측하지 못한다. 그 다음 **개입**을 세 번 시도했다: grip residual → plan residual → critic 재순위.
셋 다 teacher 를 넘지 못했고, **실패 원인이 매번 다르게 특정됐다** ([`METHOD.md`](METHOD.md) §2).
현재 위치는 **지각 기반 제어 라인의 S1 직전** — critic 을 목적함수로 계획을 직접 최적화하는 단계.

★ **2026-09-30 새벽 추가 — DIRPRED (7시간 자율 실험, 사용자 아이디어 "방향별 응답 + 신뢰도를 점군·촉각으로")**
= [`EXP_DIRPRED.md`](EXP_DIRPRED.md) / [`RESULTS_DIRPRED.md`](RESULTS_DIRPRED.md). 결론 네 줄:
**방향별 응답은 점군만으로 예측된다**(3손 홀드아웃 AUC .75~.87, A1 4/4) · **촉각은 거의 안 더한다**(B1 4/4 ❌, 최대 +.03/가림 +.06) ·
**앙상블 분산은 신뢰도가 아니다**(A2·A3 4/4 ❌) · **손을 넘지 않는다**(교차 AUC .55, 탐색적).
촉각이 살아 있는 데이터셋(Shadow/Allegro/Inspire-cc, K=5 포착)이 생겼다 → §3.

★★ **2026-09-30 저녁 추가 — HOLDPRED (사용자 "쭉 진행" 지시)** = [`EXP_HOLDPRED.md`](EXP_HOLDPRED.md) / [`RESULTS_HOLDPRED.md`](RESULTS_HOLDPRED.md).
라벨을 **파지 후 유지 상태**(hub posthold 장치, α=8, Y_pose)로 옮기고, hold 상태 예측기(H1) · 실행 전 예측기(H2) · **CEM 계획 최적화 → 물리 재생(H3)** 까지 갔다.
결론: 유지 구간은 형성 구간과 다르다 — Inspire 는 점군만으론 안 되고 손 상태 필요, Shadow 는 촉각 단독이 점군만큼 맞힌다.
**행동 이득**: 보지 않은 물체 ❌(두 손), 그러나 **Inspire 학습 물체·새 장면에서 teacher +.107 → 사전등록 복제 +.098 (CI>0, lift 유지)** —
**이 프로젝트에서 물리 재생으로 teacher 를 넘은 첫 개입(복제 포함).** 남은 벽 = 형상 일반화(학습 물체 25종).
새 자산: `runs/critic/hp_*` (posthold 라벨 데이터셋 3손 + test/replay 장면), `runs/critic/_hp/pre_*.pt` (PRE 앙상블), 스크립트 `55~57`.

## 2. 지금 당장 할 일

**S1** ([`METHOD.md`](METHOD.md) §4). 새 시뮬레이션 없이 기존 46,425 샘플로 시작할 수 있고,
GPU 는 검증 재생에만 쓴다. 착수 전에 `docs/EXP_S1.md` 로 **사전등록**할 것.

- 기준선 teacher **2.997** · 상한 고정 장면 오라클 **3.71** (같은 라벨, 교차검증됨)
- lift 예측기 동시 학습 → `카운트 s.t. lift ≥ τ` 제약 최적화
- ★ **독립 재생 검증 필수** — critic 위에서만 좋은 해를 거른다
- (DIRPRED 이후) S1 의 목적함수를 스칼라 카운트가 아니라 **6-dim 방향별 예측의 최솟값(ε)** 으로 쓸 수 있다 — `52_dirpred.py` 의 P 모델.
  단 손 입력이 없으면 손을 못 넘고, 불확실성은 축별로 정의해야 한다 (RESULTS_DIRPRED §3).
- ★ (HOLDPRED 이후) **S1 은 사실상 실행됐다** = `56_planopt.py`(PRE 앙상블 CEM) + 재생. 학습 물체에서 +.10 이 나왔으니 다음은 **형상 일반화**:
  (a) 학습 물체 수를 늘린다(ours_M/L·union_ycb 로 PRE 재학습, 홀드아웃 이득이 CI>0 이 되는가), (b) PRE 입력에 손 상태 추가,
  (c) Shadow 에서 plan 정보가 안 읽히는 원인(PRE-noplan ≈ PRE) 규명. 라벨은 α=8 Y_pose, 반복 ≥2 (Inspire corr(A,B) .42 — 3회 권장).

## 3. 데이터·산출물 (재생성 없이 쓸 수 있는 것)

| 무엇 | 어디 | 규모 |
|---|---|---|
| 버팀 카운트 라벨 데이터셋 | `runs/critic/ds_s4{2,3,4,5}/data.npz` | 46,425 후보 × 3회 측정, 장면 180 |
| 그 안의 내용 | 특권 55-d · palm PCL 256점 · 12-D 계획 · 6방향 hold ×3 · lift · 물체/장면/σ id | — |
| 판정 결과 | `results/critic_concept.{md,json}` · `results/critic_run__kimm.txt` | — |
| 선행 오라클 해 | `runs/residual/oracle_s4{2,3}/oracle.npz` | 물체별 12-D + 고정 자세 |
| ★ **촉각 살린 라벨 데이터셋** (`GS_CC=1`, K=5 지연 포착 `obsK/pclK/capK`) | `runs/critic/dp_{shadow,allegro}_s42/`, `dp_inspire_cc_s4{2,3}/data.npz` | Shadow 15,840 · Allegro 15,840(유효 9,699) · Inspire 23,760 후보, 각 ×3회 |
| DIRPRED 판정·예측값 | `results/dirpred_*__kimm.json` · `runs/critic/_dp/pred_*.npz` | 셀 4 × 포착 2 |

★ `runs/` 는 gitignore = 레포 밖 취급. **다른 머신으로 옮기려면 별도 복사**해야 한다.

## 4. 실행

```bash
# 라벨 생성 (3회 반복 측정). 1 에피소드 ≈ 59s, 264 후보/에피소드
GS_GPU=0 GS_SEED=42 GS_CC=1 bash scripts/50_dataset.sh <tag> 55
# critic 학습 + 사전등록 게이트 판정
GS_CRIT_DEV=cuda:0 $DG_PY scripts/51_critic.py runs/critic/ds_s*/data.npz
# 선행: 분기 마진 측정 / residual 학습 / 오라클 CEM
bash scripts/11_d1_driver.sh 0 runs/cells_gpu0.txt
bash scripts/41_residual.sh <mode> <tag> [iters]     # mode = min|sum|zero|plan_*|b*|sweep|oracle|replay|dataset
```

## 5. ★ 함정 (전부 실제로 당한 것)

| # | 함정 | 대응 |
|---|---|---|
| 1 | **`contact_collection: 0` = CC_NEVER** — 접촉력 18-d 가 상수 0 이다. 전 선행 실험이 촉각을 못 봤다 | `GS_CC=1` 로 켜라. 켜면 12.22 N |
| 2 | grip 진입 첫 스텝은 접촉률 **24.6%** — 촉각을 읽기엔 너무 이르다 | 특징 포착을 폐합 이후로 |
| 3 | best-of-N/CEM 의 보고값은 상한이 아니다 (편향 0.17, 자세 랜덤이면 이득을 통째로 삼킴) | 독립 재생으로만 상한을 말한다 |
| 4 | 개입 게이트를 오라클 가동범위보다 크게 잡으면 null 이 "검정력 없음"과 구분되지 않는다 | 가동범위를 먼저 재고 게이트를 정한다 |
| 5 | 다른 코드 경로의 baseline 과 비교하면 경로 차이가 효과로 둔갑한다 (가짜 양성 전례) | 같은 경로의 무개입 대조군 |
| 6 | root state 는 env-local 인데 env origin 을 더하면 상수 오차 18.0mm (=tableHeight) | 상수 오차 + 설정값 일치 → 좌표계 의심 |
| 7 | 감시 루프의 `pgrep -f <스크립트명>` 이 감시 셸 자신을 매칭해 교착 (2시간 손실) | 로그 종료 마커로 판정 |
| 8 | σ≥0.35 계획 섭동은 lift SR 이 .14 로 붕괴 | σ ≤ 0.2, 또는 lift 제약을 같이 걸어라 |

## 6. 하네스 구조 (재사용 가능)

- `num_envs = n_obj × replicas × G`, 같은 물체 env 세로줄 = 그룹.
  `G = 2 + 6·|alpha|` (main · control · 방향×α 프로브)
- `scripts/_patch/gs_branch.py` — 6방향 외력 마진 측정. reset 직후 1회 복제(replicate-once),
  `GS_BR_RESYNC_EVERY=H` 로 H 스텝마다 재동기화
- `scripts/_patch/gs_residual.py` — `PPO.run` 을 가로채 우리 루프를 돌린다.
  모드: 학습(`min`/`sum`/`b*`/`plan_*`) · 감사(`sweep`) · 오라클(`oracle`) · 재생(`replay`) · **라벨 생성(`dataset`)**
- 분기 지평 **H=5(1.67 s)** 여야 변위 라벨이 게이트를 통과한다 (외력 중엔 거의 안 움직이고 **나중에** 놓친다)

## 7. 판정 규율
[`PROTOCOL.md`](PROTOCOL.md) §5 + [`METHOD.md`](METHOD.md) §5. 요약: **임계는 수치 보기 전에 고정,
실험 조건은 바꿔도 되고 임계는 못 바꾼다. 게이트 실패는 그대로 기록한다.**
