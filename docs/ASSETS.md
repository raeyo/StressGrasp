# ASSETS — 재생성 없이 쓸 수 있는 것

> **정본 인벤토리.** 지금 이 머신에 **실재하고 바로 쓸 수 있는** 것만. 없는 것은 §4 에 **구멍**으로 표시한다.
> ★ **`runs/` 는 gitignore = 레포 밖.** `git clone` 만으로는 §1 이 따라오지 않는다 — **별도 복사.**
>
> 2026-09-30 통합: 구 `ASSETS.md` + `BASELINES.md` + `BASELINE_ACQUISITION.md` + `BASELINE_FITNESS.md`.
> ⚠ 통합 중 **RobustDexGrasp 상태가 두 문서에서 상충**했다 — §4 에 교정판을 적었다.
>
> 레포 = `git@github.com:raeyo/StressGrasp.git`
> ★ **디렉토리명 `graspstress` ≠ 레포명 `StressGrasp`** — clone 후 이름을 맞춰야 스크립트 경로가 맞는다.
>
> ★★ **이 레포는 여러 워크스페이스가 동시에 작업한다** — 규칙 [`CLAUDE.md`](../CLAUDE.md) ·
> 배정 [`CLAIMS.md`](../CLAIMS.md). **master push 는 hub 만.** 작업은 `<태그>/<주제>` 브랜치에서.
> **작업 디렉토리 하나 = 세션 하나 = 브랜치 하나** (별도 worktree, P20).
>
> | 워크스페이스 | 태그 | 역할 | 스크립트 번호 |
> |---|---|---|---|
> | hub (로컬 raeyo-pc, RTX 3080) | `hub` | **통합 담당** — master merge · 정본 문서 · 판정 · 배정 | 35–39 (postlift, 완료) |
> | kimm-h200 (git 전용, GPU 0·1) | `kimm` | 배정받은 실험 실행 · 코드 | **50–59** |
> | campus 서버 | `bengio` 등 | 배정받은 실험 실행 · 코드 | 60–69 |
>
> ⚠ **DemoGrasp 포크가 둘이다** — hub 의 `external/DemoGrasp` 는 **수정본**(`plan_anchor_pos` 등),
> kimm 은 원본 `tacdexgrasp/references/DemoGrasp`. **하니스는 두 쪽 모두에서 돌아야 한다** (P17).

---

## 1. ★ 데이터 — 가장 비싼 자산

| 무엇 | 어디 | 규모 |
|---|---|---|
| **버팀 카운트 라벨 데이터셋** | `runs/critic/ds_s4{2,3,4,5}/data.npz` | **46,425 후보 × 3회 측정**, 장면 180 (21+21+13+13 MB) |
| ↳ 내용 | 특권 **55-d** · palm **PCL 256점** · **12-D 계획** · 6방향 hold ×3 · lift · 물체/장면/σ id | |
| 선행 오라클 해 | `runs/residual/oracle_s4{2,3}/oracle.npz` | 물체별 12-D + 고정 자세 (CEM ~400회 물리탐색) |
| 촉각 켠 재측정 | `runs/critic/cc_on/` | 1 에피소드 (P7 검증용) |
| 판독 근거 요약 (커밋) | `../results/*.md` | stage0 · failmap · branch_margin · residual_arms · oracle_bound · paired · ★ **postlift__hub** |
| ★ **파지 후 다방향 측정** | `../results/postlift__hub.md` + `scripts/3{5,6}_*` | 손 3종 × α 5단 × seed 2, 방향별 R_d·R_all·e_p·e_R. **머신 = hub** |

⚠ **이 데이터셋의 접촉력 18-d 는 상수 0 이다** ([`PITFALLS.md`](PITFALLS.md) P7).
촉각을 쓰려면 **`GS_CC=1` 로 재생성**해야 한다.

★ **T1**([`EXPERIMENTS.md`](EXPERIMENTS.md) §3)과 **S1**은 **이 데이터만으로 물리 호출 0회**로 착수 가능.

## 2. 학습된 모델 — **새로 학습하지 않는다**

표기: **[공개]** = 원저자 배포본 그대로 · **[재현]** = 코드 미공개라 우리가 구현·학습.
★ **논문 표에서 이 라벨을 지우지 않는다.**

### 2-1. DemoGrasp teacher — **[공개] ckpt 7종**, `references/DemoGrasp/ckpt/`
특권 관측(GT pose + 완전 PC) · single-step MDP · **one-shot open-loop**(12-D edit 1회 → 재생).

| ckpt | embodiment | hand cfg | ref pkl | lift step |
|---|---|---|---|---|
| `inspire.pt` ★ | FR3 + Inspire RH56 (6-DoF) | (기본) | `grasp_ref_inspire.pkl` | 13 |
| `shadow.pt` / `fr3_shadow.pt` | Shadow | `shadow_simple` / `fr3_shadow` | `grasp_ref_shadow.pkl` | 11 |
| `ur5_allegro.pt` | UR5 + Allegro | `ur5_allegro` | `grasp_ref_allegro.pkl` | 11 |
| `fr3_dclaw.pt` | D'Claw | `fr3_dclaw_gripper` | `grasp_ref_dclaw_gripper.pkl` | — |
| `ur5_svh.pt` | Schunk SVH | `ur5_svh` | `grasp_ref_svh.pkl` | — |
| `fr3_panda_gripper.pt` | 2-finger gripper | `fr3_panda_gripper` | `grasp_ref_panda_gripper.pkl` | — |

★ **7종 = 손 종류 분해축을 공짜로 준다.** 2-finger gripper 는 다지 손 대비 대조군.
측정된 clean SR(`ours_S`): inspire **.762** · panda_gripper **.867** · fr3_shadow **.532** · dclaw **.262**.
⚠ **분기 마진을 잰 것은 Inspire/Shadow/Allegro 3종뿐**이다 (나머지는 원저자 recipe 가 달라 별도 인자 필요).

### 2-2. DemoGrasp student (GR00T) — **[재현] 8종**, `demograsp_repro/student/runs/`
RGB 2뷰(256²)+proprio → 13-D action. 논문 스펙(사전학습 ViT + flow-matching head)을 우리가 구현·학습.

| run | 손 | 모드 |
|---|---|---|
| `groot_v1` | Inspire | open-loop (논문 student, 재현 확인됨) |
| `groot_shadow` / `groot_allegro` | Shadow / Allegro | open-loop |
| `groot_ft_dr` | Inspire | ViT finetune + 도메인 랜덤화 (3.6 GB) |
| **`groot_cl_shadow_K4/K8/K16`** | Shadow | **closed-loop (receding horizon)** |
| `v0` | Inspire | scratch ResNet18 (저성능 대조) |

★★ **`eval_student_groot.py` 의 `exec_H` 로 하나의 ckpt 를 open/closed 둘 다 평가한다**
(`exec_H=0` → open-loop, `8/4/2` → closed-loop). 가중치·구조·데이터가 **전부 같고 반응성만 다르다**
→ "반응하면 외란을 이기는가"의 **가장 깨끗한 대조**.
★ Shadow 에 사다리 전체가 있다: `shadow.pt`(특권·open) → `groot_shadow`(RGB·open) → `groot_cl_shadow_K*`(RGB·closed).

⚠ **이 자산은 아직 한 번도 쓰이지 않았다.** 2026-09-22 에 "주 baseline = `groot_cl_shadow_K8`" 로
결정됐으나, 이후 전 실험(분기·residual·critic)은 **teacher `inspire.pt` 로만** 돌았다.
open-loop vs closed-loop 축은 **미실행**이다.

### 2-3. UniGraspTransformer — **[공개] env·코드 / [재현] 정책**
`external/UniGraspTransformer` + `phasegrasp/benchmark/` ckpt 5종
(state universal `ugt_state_full3200_model_best.pt` · vision `v2_base/model_100.pt` 외).
★ **다른 env · 다른 성공 판정**(lift 0.05 + hold) → DemoGrasp 수치와 **직접 비교 금지.**
"패턴이 env 를 바꿔도 재현되는가"의 **교차 확인용만**.

### 2-4. 비교 가능성 규칙
| 묶음 | 직접 비교 | 이유 |
|---|---|---|
| teacher 7종 끼리 | ✅ | 같은 env·판정·물체셋 |
| teacher vs GR00T student | ✅ (**열 분리**) | 같은 env·판정. 단 teacher 는 **특권 관측** |
| GR00T `exec_H=0` vs `8/4` | ✅✅ **가장 깨끗** | 같은 가중치, 반응성만 다름 |
| UGT | ❌ | 다른 env·판정 |

## 3. 하네스 — 재사용 가능한 측정 장치

`scripts/_patch/` = 남의 코드에 **런타임으로 얹는** 우리 패치. ★ `references/DemoGrasp` 는 고치지 않는다.

| 파일 | 하는 일 |
|---|---|
| `gs_branch.py` | **6방향 외력 마진 측정.** replicate-once / per-step(H) 두 모드, `GS_BR_FRAME`(world/palm), `GS_BR_RESYNC_EVERY=H` |
| `gs_residual.py` | `PPO.run` 을 가로채 우리 루프를 돌린다. 모드 = 학습(`min`/`sum`/`b*`/`plan_*`) · 감사(`sweep`) · 오라클(`oracle`) · 재생(`replay`) · **라벨 생성(`dataset`)** |
| `gs_stress.py` | D2/D3 스트레스 주입 (질량 배율 · 외력) |
| `usercustomize.py` | 패치 자동 로드 |

구조·파라미터 = [`PROTOCOL.md`](PROTOCOL.md) §4·§7.

★ **이 하네스 자체가 기여 후보다** — [`METHOD.md`](METHOD.md) §2 ① 이 "기존 연구가 안 한 이유 중
하나가 측정 비용·시뮬 구조"라고 진단했고, 우리는 결함 7건을 지나 그걸 해결했다.
단 **D1(측정 논문 안 한다) 때문에 단독 기여로는 못 쓰고** 해결책 논문의 **인프라 절**로만.

## 4. ⚠ 구멍 — 없는 것

| 무엇 | 상태 | 왜 아픈가 |
|---|---|---|
| **RobustDexGrasp** | ★ **포팅은 되어 있다** — `demograsp_repro/robustdex/eval_robustdex.py`(+`rdx_policy.py`), 훅 `run_rl_grasp.py +debug=eval_robustdex`, RaiSim 원본 대조 기준(SR **.969**), 공정 물체셋 `ycb_common19`. **RaiSim 런타임은 필요 없다.** 없는 것은 **`RDX_ROOT`(원본 레포)** — ckpt `full_5500_r.pt` + helper `initial_pose_final.py`·`inverseKinematicsUR5.py`. 레포 URL 또는 hub 경로가 필요하다 | **외란 강건성을 정면으로 주장하는 유일한 공개 정책**이자 유일한 직접 비교군 |
| **촉각이 살아 있는 라벨 데이터셋** | 없음 | §1 의 46,425 는 접촉력이 0. 촉각 실험은 **재생성부터** |
| **가려진 점군** | 없음 | 증류 라인 실험 비용의 대부분이 여기 |
| **Allegro 의 복제 잡음** | 게이트 실패 원인 **미조사** (control Y_hold .95~.96 < .97) | E11 에서 Allegro 셀이 통째로 무효다. grip 이 짧다는 기존 관측과 관련 가능 |
| **형성 단계 Shadow·Allegro 마진 재측정** | 필요 (P16) | [`FACTS.md`](FACTS.md) **A2** 의 근거가 인용 보류 상태다 |
| **kimm 원본 DemoGrasp 에서의 postlift 재현** | 미확인 | E11 은 hub 의 **수정본** 포크에서 쟀다 (P17) |

⚠ **정정 기록**: 구 `BASELINES.md` §3 은 RobustDexGrasp 를 *"RaiSim → IsaacGym 포팅 미완"* 이라고
적었으나 **틀렸다.** 포팅은 완료돼 있고 실제로 돌린 기록도 있다
([`EXPERIMENTS.md`](EXPERIMENTS.md) X3 = RDX 평가에서 나온 무효 셀). 막힌 것은 **원본 레포 경로**다.

## 5. 실행

```bash
cd $VITAC_ROOT/projects/graspstress && source scripts/_common.sh && gs_check
# python = $DG_PY ($HOME/miniconda3/envs/demograsp, py3.8) — conda activate 금지, 절대경로 실행

# 라벨 생성 (3회 반복). 1 에피소드 ≈ 59 s, 264 후보/에피소드.  ★ GS_CC=1 없으면 촉각이 죽는다
GS_GPU=0 GS_SEED=42 GS_CC=1 bash scripts/50_dataset.sh <tag> 55
# critic 학습 + 게이트 판정
GS_CRIT_DEV=cuda:0 $DG_PY scripts/51_critic.py runs/critic/ds_s*/data.npz
# 분기 마진 / residual 학습 / 오라클 CEM / 감사 / 재생
bash scripts/11_d1_driver.sh 0 runs/cells_gpu0.txt
bash scripts/41_residual.sh <mode> <tag> [iters]   # min|sum|zero|plan_*|b*|sweep|oracle|replay|dataset
# D1 유지시간 sweep 한 셀 (코드 변경 0 — episodeLength 오버라이드만)
GS_GPU=0 bash scripts/10_hold_sweep.sh ckpt/inspire.pt ours_bench/ours_S.yaml 140 42 66
python3 scripts/20_summarize.py       # 집계
```
★ 감시 루프에 **`pgrep -f` 금지** (P13) — 로그 종료 마커로 판정한다.
