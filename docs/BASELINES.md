# BASELINES — 가져다 쓸 학습된 모델 대장

> 원칙: **새로 학습하지 않는다.** 이 단계는 "기존 방법들로 이 문제를 최대한 풀어본다"이다.
> 표기 규약: **[공개]** = 원저자가 배포한 것을 그대로 실행 · **[재현]** = 코드 미공개라 우리가 구현·학습.
> 논문 표에서 이 라벨을 지우지 않는다.

## 1. DemoGrasp (ICLR'26, arXiv 2509.22149, MIT)

### 1.1 Teacher — **[공개]** ckpt 7종, `references/DemoGrasp/ckpt/`
특권 state 관측(GT 물체 pose + 완전 PC) · single-step MDP · **12-D edit 을 한 번 내고 open-loop 재생**.

| ckpt | embodiment | 크기 |
|---|---|---|
| `inspire.pt` | FR3 + Inspire RH56 (6-DoF hand) | 16.4 MB |
| `shadow.pt` / `fr3_shadow.pt` | Shadow Hand | 16.5 MB |
| `ur5_allegro.pt` | UR5 + Allegro | 16.5 MB |
| `fr3_dclaw.pt` | D'Claw | 16.4 MB |
| `ur5_svh.pt` | Schunk SVH | 16.4 MB |
| `fr3_panda_gripper.pt` | 2-finger gripper (대조군) | 16.4 MB |

★ **7종 embodiment = RQ1 의 "손 종류" 분해축**을 공짜로 준다. 2-finger gripper 는 다지 손 대비 대조군.

### 1.2 Student — **[재현]** (원저자 student 코드·ckpt 미공개)
논문 스펙(GR00T-N1.5 = 사전학습 ViT + flow-matching action head)을 우리가 구현·학습.
RGB 2뷰(256²)+proprio → 13-D action, **one-shot**(첫 프레임 1회 관측 → 계획 → open-loop 재생).

| run | 내용 | 비고 |
|---|---|---|
| `groot_v1` | frozen SigLIP-2 SO400M + FM DiT head, Inspire | 논문 재현 확인됨 |
| `groot_shadow` / `groot_allegro` | 동일 recipe, 다른 손 | |
| `groot_ft_dr` | ViT finetune + 도메인 랜덤화 (3.6 GB) | |
| **`groot_cl_shadow_K4/K8/K16`** | ★ **closed-loop 변형** (K 스텝마다 재계획) | **RQ4 의 open vs closed 축** |
| `v0` | scratch ResNet18 | 저성능 대조 |

★ `groot_cl_*` 가 있어서 **RQ4(open-loop vs closed-loop 낙폭 차이)를 새 학습 없이** 물을 수 있다.

## 2. UniGraspTransformer (UGT, MIT) — **[공개] env·코드 / [재현] 정책**
원저자 ckpt 가 아니라 **우리가 재학습한** 정책이다. 절대값을 논문값과 비교하지 않는다.

| ckpt | 내용 |
|---|---|
| `phasegrasp/benchmark/shared_ckpts/ugt_state_full3200_model_best.pt` | state(특권) universal student |
| `phasegrasp/benchmark/git_share/v2_base/model_100.pt` | vision universal student |

★ 지위: **2차 확인용**. 다른 env·다른 성공 판정이라 DemoGrasp 수치와 **직접 비교 금지**.
"낙폭이 DemoGrasp 고유 현상인가"를 교차 확인하는 데만 쓴다.

## 3. RobustDexGrasp (2504.05287) — **미확보 (구멍)**
공개 ckpt(`full_5500_r.pt`)는 있고 네트워크 로드도 검증됐으나 **RaiSim → IsaacGym 포팅 미완**.
closed-loop robust 정책의 대표라 **RQ4 의 가장 강한 대조군인데 지금 없다.**
→ 1차 측정에는 넣지 않는다. D1/D2/D3 에서 문제가 실재로 판정되면 그때 포팅을 정식 과제로 올린다.

## 4. 착수 순서

1. **D1 × teacher(inspire) × ours_S/M/L** — 코드 변경 0, 가장 싼 스크리닝
2. D1 × teacher 7종 — embodiment 분해
3. D2/D3 × teacher — 하중·외력 (패치 필요)
4. D1~D3 × student(one-shot) vs student(closed-loop K4/K8/K16) — RQ4
5. (조건부) UGT 교차 확인

## 5. 공정성 각주 (논문에 반드시)
- teacher 는 **특권 관측**이다. student 와 같은 표에 두되 열을 분리한다.
- student 는 **[재현]** 이다. 원저자 구현과 다를 수 있다.
- DemoGrasp 성공 판정(`delta_z>0.1`, hold 없음)과 다른 벤치의 판정은 호환되지 않는다.
