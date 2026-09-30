# EXP — Amortized actor + 온라인 반복 (ACTOR) · 사전등록 (2026-09-30 밤)

> ★ 판정 규칙은 수치 보기 전 고정 (PROTOCOL §5). 선행 = [`RESULTS_HOLDPRED.md`](RESULTS_HOLDPRED.md) (PRE critic + CEM: Inspire 학습 물체·새 장면 +.107/+.098, 보지 않은 물체 ✗).
> 사용자 착상(2026-09-30 밤): *"actor-critic 에서 critic 의 출력을 actor 가 유추하면서 학습하는 것 아닌가 — end-to-end 로."*
> 워크스페이스 kimm. EXP/RESULTS 는 규칙상 hub 정본 — 사용자 지시로 kimm 작성, hub 검토 요청.

## 0. 한 줄
> DemoGrasp 는 one-shot 이라 문제가 contextual bandit 이다. PRE(장면, 계획) 가 곧 Q(s,a). **① CEM(비모수 argmax)을 신경망 actor π(s) 로 amortize** 하면
> 같은 이득이 나는가. **② actor 가 낸 계획을 물리로 라벨링해 critic 을 다시 맞추는 온라인 반복**이 critic 의 계획 민감도와 이득을 올리는가.

## 1. 방법 (동결)
- **actor**: 입력 = critic 앙상블 0번의 PointNet 임베딩(동결) + 물체 자세 q + teacher edit → MLP(256-128) → Δ(12). 계획 = clamp(teach + 0.3·tanh(Δ), −1, 1) (신뢰영역 ±0.3, CEM 과 동일).
- **목적**: 비관적 Q. 앙상블 5개 각각의 `min_d p_pose(d)` 의 **최솟값** + g0 제약 `−2·relu(0.5 − min_k g0_k)` − 0.1·‖Δ‖²/12. Adam 1e-3, 300 epoch, 학습 컨텍스트 = 학습 데이터의 (장면, 물체) 슬롯0 행(홀드아웃 물체 포함 — 라벨 아닌 장면 특징만 쓰므로 누수 아님; 단 판정은 test 장면).
- **평가**: test 장면 seed 99 (HOLDPRED 와 동일 12 장면) 에 슬롯0 = teacher, **슬롯1 = actor**, 슬롯2~7 = actor + σ∈{.05,.05,.1,.1,.2,.2} 잡음. 물리 재생 2회. `57_gain.py` (opt = 슬롯1).
- **온라인 반복** (R=4): 라운드마다 학습 장면 15개(s42 의 p0 재사용)에 actor 계획(+잡음 6개)을 재생(2회) → 누적 데이터로 PRE 재학습(앙상블 5) → actor 재학습 → 다음 라운드. 마지막에 test 장면 재생.
- 손 = Inspire (GPU 0) · Shadow (GPU 1). Allegro 는 critic 이 약해(ρ .40) 제외.

## 2. ★ 게이트 (수치 보기 전 고정)
| 게이트 | 묻는 것 | 단위 | 기준 |
|---|---|---|---|
| **A1** | amortized actor 가 CEM 만큼 이득을 내는가 | 전 물체 396 (HOLDPRED §1-6 과 동일) | actor − teacher ≥ **+.05**, 페어드 부트스트랩 95% CI > 0, g0 Δ ≥ −.05 |
| A1′ | CEM 대비 | 〃 | actor − CEM 병기 (판정 아님). Inspire CEM = +.107 (같은 장면) |
| **A2** | 온라인 반복이 이득을 키우는가 | 전 물체 396, test 장면 | (actor_R − teacher) − (actor_0 − teacher) ≥ **+.03** 이고 actor_R CI > 0 |
| A3 | 보지 않은 물체로 넘어가는가 | 홀드아웃 물체 96 | actor_R − teacher 의 CI > 0 (통과하면 최초. 기대치 낮음 — 병기) |
| A4 | critic 계획 민감도 | 홀드아웃 | PRE_R 의 `PRE − PRE-noplan` AUC 차 ≥ +.03 (Shadow 는 현재 .00) |
- Shadow 는 A1 기준을 같은 +.05 로 (CEM 이 +.034 n.s. 였으므로 통과하면 amortize 가 CEM 보다 낫다는 뜻).
- 불통과 시: A1 ✗ → actor 가 critic 오류를 파고든 것(예측 pmin vs 실측 상관으로 확인). A2 ✗ → 온라인 라벨이 critic 을 못 고침(라운드별 critic ρ 병기).

## 3. 병기 필수
라운드별: 재생 R_all_pose(actor, teacher) · PRE stage2 ρ · ‖Δ‖ 분포 · 예측 pmin(actor) vs 실측. 최종: actor_0 vs actor_R vs CEM 표.

## 4. 말하지 않는 것
teacher 재학습(T3) · 형상 확대 · 폐루프 · 실물.

## 5. 실행 기록

### 5-1. Round 0 (amortized actor, critic 고정) — 2026-10-01 00:30
| 손 | 전 물체 actor − teacher [CI] | g0 Δ | 홀드아웃 물체 [CI] | CEM (같은 장면) | A1 |
|---|---|---|---|---|---|
| Inspire | **+.085 [+.048, +.121]** | −.003 | −.005 [−.089, +.078] | +.107 | ✅ (CEM 의 ~80%, 탐색 없이) |
| **Shadow** | **+.066 [+.028, +.104]** | +.018 | **+.089 [+.036, +.146]** (g0 +.094) | +.031 n.s. | ✅ — **CEM 보다 낫고, 홀드아웃 물체 CI>0 최초** |
actor 이동량 ‖Δ‖ ≈ .31 (CEM .5~.75). 비관적(앙상블 min) 목적 + 작은 이동이 CEM 의 공격적 탐색보다 물리에서 낫다.
사용자 결정(00:40): **이후 PoC 는 Shadow 단일 손으로.** Inspire 체인은 대조용으로만 끝까지 두고, Allegro 는 접는다.

### 5-2. 최종 판정 (온라인 4라운드 후, 2026-10-01 02:00) — 판독 = RESULTS_ACTOR.md
| 손 | actor_0 전 물체 | actor_R 전 물체 | A2 (≥ +.03) | actor_R 홀드아웃 | A3 (CI>0) | A4 plan 기여 |
|---|---|---|---|---|---|---|
| **Shadow** | +.066 | **+.109 [+.073, +.144]** | ✅ +.043 | **+.094 [+.031, +.156]** | ✅ **최초** | ❌ .005 (r1~r4: .007/.016/.021/.005) |
| Inspire | +.085 | +.097 [+.059, +.136] | ❌ +.012 | +.042 [−.031, +.115] | ❌ | ✅ .066 (원래 있음) |
g0: Shadow +.053(전)/+.141(홀드아웃), Inspire +.021/+.031. GPU 0·1 반환. 라운드당 재생 15 장면 ≈ Inspire 13 분 / Shadow 17 분.
