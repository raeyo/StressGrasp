# BASELINE 확보 상태 (2026-09-22)

> 목적 = "언제 잘 안 되는지"를 재려면 **비교 가능한 baseline 이 여러 개** 있어야 한다.
> 판정은 여기서 내리지 않는다. 확보 상태와 막힌 지점만 적는다.

## 1. 확보 완료 — 이 머신에서 바로 돈다

### DemoGrasp teacher — **[공개 ckpt 7종]**, `references/DemoGrasp/ckpt/`
특권 관측(GT pose + 완전 PC) · single-step MDP · **one-shot open-loop**.

| ckpt | embodiment | hand cfg | ref pkl | lift step |
|---|---|---|---|---|
| `inspire.pt` | FR3 + Inspire (6-DoF) | (기본) | `grasp_ref_inspire.pkl` | 13 |
| `shadow.pt` | Shadow | `shadow_simple` | `grasp_ref_shadow.pkl` | 11 |
| `fr3_shadow.pt` | FR3 + Shadow | `fr3_shadow` | `grasp_ref_shadow.pkl` | 11 |
| `ur5_allegro.pt` | UR5 + Allegro | `ur5_allegro` | `grasp_ref_allegro.pkl` | 11 |
| `fr3_dclaw.pt` | D'Claw | `fr3_dclaw_gripper` | `grasp_ref_dclaw_gripper.pkl` | — |
| `ur5_svh.pt` | Schunk SVH | `ur5_svh` | `grasp_ref_svh.pkl` | — |
| `fr3_panda_gripper.pt` | 2-finger gripper | `fr3_panda_gripper` | `grasp_ref_panda_gripper.pkl` | — |

★ **2-finger gripper 가 대조군으로 유용하다** — 다지 손의 이점이 어디서 나오는지 가른다.

### DemoGrasp student (GR00T) — **[재현 8종]**, `demograsp_repro/student/runs/`
| run | 손 | 모드 |
|---|---|---|
| `groot_v1` | Inspire | open-loop (논문 student) |
| `groot_shadow` / `groot_allegro` | Shadow / Allegro | open-loop |
| `groot_ft_dr` | Inspire | ViT finetune + 도메인 랜덤화 |
| **`groot_cl_shadow_K4/K8/K16`** | Shadow | **closed-loop (receding horizon)** |
| `v0` | Inspire | scratch ResNet18 (저성능 대조) |

★★ `eval_student_groot.py` 의 `exec_H` 로 **하나의 ckpt 를 open/closed 둘 다** 평가한다
(`exec_H=0` → open-loop, `exec_H=8/4/2` → closed-loop). 가중치·구조·데이터가 동일하고
**반응성만** 다르다 → "반응하면 외란을 이기는가"의 가장 깨끗한 대조.

### UniGraspTransformer — **[공개 env·코드 / 재현 정책]**
`external/UniGraspTransformer` + `projects/phasegrasp/benchmark/` ckpt 5종
(`v2_base/model_100.pt` vision · `shared_ckpts/ugt_state_full3200_model_best.pt` state 외).
★ **다른 env · 다른 성공 판정**(lift 0.05 + hold) → DemoGrasp 수치와 **직접 비교 금지**.
"낙폭 패턴이 env 를 바꿔도 재현되는가"의 교차 확인용.

## 2. ★ 막힘 — RobustDexGrasp (closed-loop robust 대표)

**포팅은 이미 되어 있다** (문서 `ROBUSTDEXGRASP_PORT.md` 의 "계획" 서술은 낡았다):

| 있는 것 | 위치 |
|---|---|
| IsaacGym 포팅 eval | `demograsp_repro/robustdex/eval_robustdex.py` (+ `rdx_policy.py`) |
| 실행 훅 | `run_rl_grasp.py` `+debug=eval_robustdex` |
| **RaiSim 원본 대조 기준** | `robustdex/raisim_baseline/` — 원본 SR **0.969**, `|action|` 4.68 실측 |
| **공정 비교 물체셋** | `raisim_baseline/ycb_common19.{txt,yaml}` — RaiSim·DemoGrasp 양쪽에 있는 YCB 19종 |
| 프레임 대조 데이터 | `robustdex/raisim_gt/gt_mug_theta0.npz` |

**없는 것 = `RDX_ROOT`**: RobustDexGrasp 원본 레포. 다음이 그 아래 있어야 한다.
- ckpt `raisimGymTorch/data_all/student/student_ckpt/full_5500_r.pt`
- helper `initial_pose_final.py` (pre-grasp 자세 샘플러) · `inverseKinematicsUR5.py` (UR5 해석 IK)

★ 이 머신에 레포가 없고, GitHub 레포명 추측 6종 전부 실패 (ssh 인증은 정상).
**필요한 것: 레포 URL 또는 hub(raeyo-pc) 경로.** 확보되면
`external/RobustDexGrasp` 에 **실물로** 두고 (우산 README §1 — 남의 코드는 심볼릭 금지)
`RDX_ROOT` 로 가리킨다. RaiSim 런타임은 **필요 없다** — 포팅본은 IsaacGym 에서 돈다.

## 3. 비교 가능성 규칙

| 묶음 | 직접 비교 | 이유 |
|---|---|---|
| DemoGrasp teacher 7종 끼리 | ✅ | 같은 env·판정·물체셋 |
| teacher vs GR00T student | ✅ (열 분리) | 같은 env·판정. 단 teacher 는 **특권 관측** |
| GR00T `exec_H=0` vs `H=8/4` | ✅✅ **가장 깨끗** | 같은 가중치, 반응성만 다름 |
| RobustDexGrasp (포팅본) | ✅ (확보 시) | 같은 env. 단 **원본 SR 0.969 재현 확인이 선행**돼야 함 |
| UGT | ❌ | 다른 env·판정 → 패턴만 교차확인 |
