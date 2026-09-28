# EXP — 분기 마진 M(s_t) 측정 · 사전등록

> ★ **판정 규칙은 측정 전에 고정한다.** 이 문서 §4·§5 는 첫 수치를 보기 전에 작성됐다 (2026-09-28).
> 수치를 본 뒤 임계값을 바꾸면 그 판정은 무효다. (graspstress PROTOCOL §5 규율 승계)
> 상위 문제 = [`PROBLEM.md`](PROBLEM.md) — 단 Stage 0 이 "파지 후 유지"를 기각했으므로
> 이 실험은 **파지 형성(grip) 단계의 외력 마진**으로 축을 옮긴다 (`FAILURE_MAP.md` §2 근거).

## 0. 목표 — 한 줄

> DemoGrasp 파이프라인이 만드는 grasp 에 대해, grip phase 의 **매 스텝**에서
> **±x,±y,±z 6방향 외력을 견디는 마진 M(s_t)** 을 물리로 측정한다.

용도 두 가지: ① **기존 방법의 한계를 수치화** (lift 성공한 grasp 들 사이 M 의 분산)
② 이후 residual RL 의 **보상 신호**. 이 문서는 ①까지만 다룬다.

## 1. 측정 원리 — 그림자 env 그룹

DemoGrasp 는 env i 에 물체 `i % n_obj` 를 배정한다 (`tasks/grasp.py:399`). 따라서
`num_envs = n_obj × G` 로 잡으면 **env {j, j+n_obj, j+2·n_obj, …} 는 같은 물체**다.
이 세로줄을 한 **그룹**으로 쓴다.

| 그룹 g | 역할 | 외력 |
|---|---|---|
| 0 | **main** — 정책이 실제로 구동 | 없음 |
| 1 | **control** — main 상태를 복사만 함 | **0 (무교란)** |
| 2 … | **probe** — main 상태 복사 + 외력 | (dir, α) 조합 하나씩 |

grip phase 의 매 env-step 마다 main → 그림자로 상태를 복사하고, decimation(20) substep 동안
외력을 걸며, 지정 substep 에서 **main 대비 물체 변위**를 잰다.

**변위를 main 기준으로 재는 것이 핵심이다** — grip 중 손가락이 물체를 미는 정상 운동이
자동으로 상쇄된다. 즉 group 1(control)의 변위 = **순수 복원 오차**이고, 그것이 곧 §4 의 게이트다.

### 복사하는 상태 (빠지면 브랜치가 거짓말을 한다)
1. 물체 root state 13종 (pos·quat·linvel·angvel) — ★ **env origin 보정 후** (envSpacing=1.0 이라 env 마다 world 좌표가 다르다)
2. robot dof state (pos·vel)
3. **`prev_targets` · `cur_targets`** — ★ PD 목표. 빠지면 그림자 손이 다른 목표로 풀려 물체를 놓는다
4. `progress_buf`
5. (reset 시 1회) `table_heights` + table root state — 테이블 높이가 env 마다 랜덤이라 안 맞추면 물체가 뜬다/박힌다

### 외력
- 크기 = `α · m · g` (물체 질량 정규화 — 물체별 난이도 왜곡 방지, RESULTS_DIGEST §4 규율)
- ★ **`simulate()` 직전마다 재적용** — `apply_rigid_body_force_tensors` 는 다음 simulate 1회에만
  걸린다. 한 번만 걸면 의도한 힘의 1/20 이 된다 (2026-09-22 실측 결함, `gs_stress.py` 와 동일 처리)
- 방향 = env/world 축 ±x,±y,±z 6개. **토크는 넣지 않는다** (v1 한계로 명시)

## 2. 시점 정의
성공 판정식(`reward.py:reward_binary`)이 쓰는 값을 그대로 재계산한다. 별도 기준을 만들지 않는다.
```
flag   = (fingertips_dist <= 0.12*n_fingers) OR (palm_dist <= 0.15)
lifted = object_delta_z > 0.1
grip   = flag AND (NOT lifted)          ← 이 실험의 측정 구간
```

## 3. 셀
| | 값 |
|---|---|
| 정책 | `ckpt/inspire.pt` (DemoGrasp teacher, 공개) |
| 물체셋 | `ours_bench/ours_S.yaml` (33 물체, 3–6 cm) — Stage 0 이 가장 많이 잰 셋 |
| α 사다리 | 1 · 2 · 5 (무게 배수) |
| 방향 | ±x ±y ±z (6) |
| 스냅샷 | substep k = 1 · 2 · 5 · 10 · 20 (1 substep = 16.67 ms) |
| seed | 42 |

- **S0 (게이트)**: G=2 (main + control). `num_envs = 66`.
- **S1 (측정)**: G = 2 + 6×3 = 20. `num_envs = 660`.

## 4. ★ S0 판정 규칙 (측정 전 고정) — 브랜치 충실도

control 그룹(외력 0)의 main 대비 변위 `drift(k)` 를, 같은 k 에서의 **α=1 probe 변위** `sig(k)` 와 비교한다.

| 관측 | 판정 |
|---|---|
| `median drift(k=5) / median sig(k=5) < 0.20` | **통과** — 브랜치 하니스 채택 |
| 0.20 ~ 0.50 | **유보** — 복사 항목 누락 탐색 후 재측정 |
| ≥ 0.50 | **실패** — 상태복원 설계 폐기, 폴백(롤아웃 중 시점 교란 스케줄)으로 전환 |

추가 무효 조건: control 변위가 물체 지름(3–6 cm)의 10% 를 넘으면 그 셀 무효 (복원이 깨진 것).

## 5. ★ S1 판정 규칙 (측정 전 고정) — 기존 방법의 한계가 실재하는가

`M(s_t)` = 6방향 **전부** 변위 < θ 인 최대 α. θ 는 물체 지름의 20% 로 정하고 k=5 에서 읽는다.
에피소드 단위 요약 `M_ep = median_t M(s_t)` (grip phase 전체).

| 질문 | 통계 | 판정 |
|---|---|---|
| **Q1. lift 성공자 사이에 마진이 퍼져 있는가** | 최종 성공 에피소드만 모아 `M_ep` 분포 | IQR ≥ 사다리 1칸 → **퍼져 있다 = lift SR 이 안정성 분산을 숨긴다** |
| **Q2. 마진이 결과를 예측하는가** | `M_ep` → 최종 성공 여부 AUC (물체 단위 층화) | AUC ≥ 0.65 → 보상 신호로 쓸 자격 있음 |
| **Q3. 방향 이방성이 있는가** | 방향별 변위 중앙값 | 특정 축이 2배 이상 약하면 기록 |

**F-기준**: Q1 에서 IQR = 0 (전부 같은 마진)이면 → residual RL 로 올릴 여지가 없다.
**S2(학습) 진행하지 않는다.** Q2 AUC < 0.6 이면 → 이 마진은 결과와 무관하므로 보상으로 부적격.

## 5-1. 실행 결과 (2026-09-29 추가)

측정 완료. 판독 = [`RESULTS_BRANCH.md`](RESULTS_BRANCH.md) · 수치 = [`../results/branch_margin.md`](../results/branch_margin.md).

★ **설계 변경 기록 (정직성)**: §1 의 "grip 매 스텝 상태 복사"는 첫 구현에서 게이트 실패했고,
원인이 구현 버그(env origin)로 밝혀져 수정됐다. 그 사이에 **복제 시점을 리셋 직후로 옮긴
replicate-once 모드**를 추가로 만들었고, 본 결과(§3)는 그 모드로 냈다. 그 모드는 §4 의
변위 비율 게이트가 적용되지 않으므로(변위가 아니라 **생존**을 읽는다) 별도 게이트
(control 그룹이 main 의 성공을 재현하는가)를 썼다 — 이것은 **수치를 보기 전에** 정의했다.
per-step 모드는 §4 의 원 게이트를 그대로 적용해 **유보** 판정을 받았고, 그대로 기록한다.

## 5-2. 추가 실험 (2026-09-29)

사전등록 §3 셀에 더해 다음을 돌렸다. 모두 **판정 규칙은 §4·§5 를 그대로 적용**했다.

| 추가 | 왜 | 결과 위치 |
|---|---|---|
| **외력 좌표계 palm** (`GS_BR_FRAME=palm`) | world 축은 손 자세와 무관해 약한 축을 비껴갈 수 있다 | RESULTS §3-1·3-2 |
| **분기 지평 H** (`GS_BR_RESYNC_EVERY`) | H=1 변위 라벨이 §4 게이트에서 유보 → 지평만 늘려 재측정 | RESULTS §2-1 |
| **embodiment 3종** (Inspire/Shadow/Allegro) | lift SR 과 마진의 순위가 같은가 (graspstress PROBLEM RQ2) | RESULTS §3-3 |
| **물체 크기** (`ours_L`) | 이방성이 물체 크기에 의존하는가 | RESULTS §3-3 |

★ 임계값은 사후 변경하지 않았다. H 연장은 **임계가 아니라 실험 조건**을 바꾼 것이다.

## 6. 기록
- 원시 = `runs/branch/<tag>/branch.npz` (레포 밖 취급)
- 요약 = `results/branch_*.md` (커밋)
- GPU = **0·1 만** (공용 서버)
