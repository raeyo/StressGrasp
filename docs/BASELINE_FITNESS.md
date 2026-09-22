# BASELINE 적합성 판정 — 이 알고리즘으로 "외란" 질문을 물을 수 있는가

> 2026-09-22. Stage 0 을 한 번 돌리고 나서야 드러난 문제를 정리한다.
> **판정 기준**: 외란 연구의 baseline 이 되려면 **외란을 관측하고 행동을 바꿀 수 있어야** 한다.
> 그럴 수 없는 정책에서 나온 "낙폭 없음"은 **정책의 강건성이 아니라 기계적 수동 안정성**을 잰 것이다.

## 1. Stage 0 의 설계 결함 (자기 지적)

Stage 0 은 **DemoGrasp teacher `inspire.pt` 하나로만** 측정했다. 그 정책은:

- **특권 관측** — GT 물체 pose + 완전 point cloud (1550-d). 실제 센서로 얻을 수 없다.
- **one-shot open-loop** — t=0 에 12-D edit 을 한 번 내면 env 가 궤적을 합성해 **재생**한다.
  중간에 무슨 일이 일어나도 **정책은 그것을 보지도, 반응하지도 않는다.**
- 레퍼런스가 끝나면 손 관절 목표가 **고정**되어 위치제어 PD 가 물체를 사실상 케이지에 가둔다.

→ 따라서 Stage 0 의 "파지 후 외란에 낙폭 없음"은
**"이 파지가 강건하다"가 아니라 "고정된 손 자세의 수동 기계적 안정성이 충분하다"** 만 말한다.
**외란 연구의 baseline 으로는 불충분하다.** 이것이 "정리가 안 되는" 원인이었다.

## 2. 후보별 적합성

| 정책 | 관측 | 제어 구조 | 외란에 반응 가능? | 적합성 |
|---|---|---|---|---|
| **DemoGrasp teacher** (`inspire.pt` 외 6종) | 특권 (GT pose + 완전 PC) | one-shot open-loop | ❌ 구조적으로 불가 | ▲ **하한선 대조군으로만** |
| **groot_v1 / groot_shadow / groot_allegro** | RGB 2뷰 **1회** | one-shot open-loop | ❌ 불가 | ▲ 동일 (단 관측은 현실적) |
| **groot_cl_shadow_K4/K8/K16** | RGB **매 H 스텝 재관측** | **closed-loop receding horizon** | ✅ **가능** | ★ **적합 — 주 baseline** |
| RobustDexGrasp | 초기 1회 vision + proprio/contact | closed-loop 5 Hz | ✅ | ★ 적합하나 **미확보**(포팅 미완) |
| UGT student (phasegrasp) | per-step | closed-loop | ✅ | ○ **다른 env·다른 판정** — 교차확인용만 |

## 3. ★ 핵심 설계 — 같은 가중치로 반응성만 분리한다

`eval_student_groot.py` 는 **하나의 ckpt** 를 두 모드로 평가한다 (`exec_H`):

```
exec_H = 0            → OPEN-LOOP : frame-0 관측 → K-step 계획 → 전부 재생   (논문 student)
exec_H = 8 / 4 / 2    → CLOSED-LOOP: H 스텝마다 재관측 → 새 chunk → 앞 H 실행
```

따라서 **`groot_cl_shadow_K8` 한 ckpt 로 open-loop vs closed-loop 를 비교하면
아키텍처·데이터·가중치가 전부 같고 오직 반응성만 다르다.** 이것이 "외란에 반응하는 것이
실제로 도움이 되는가"를 묻는 **가장 깨끗한 대조**다.

★ embodiment 정합: Shadow 손에 세 단계가 모두 있다 —
`shadow.pt`(특권 open-loop) · `groot_shadow`(RGB open-loop) · `groot_cl_shadow_K*`(RGB closed-loop).
**같은 손에서 사다리 전체를 비교할 수 있다.**

## 4. 판정

| | 결정 |
|---|---|
| **주 baseline** | `groot_cl_shadow_K8` — `exec_H` 로 open/closed 대조 |
| **하한선 대조군** | `shadow.pt` teacher (특권·open-loop) = "반응이 전혀 없을 때의 바닥" |
| **상한선 참조** | teacher 가 특권 관측을 쓰므로 clean SR 의 상한 역할 |
| **제외** | UGT (다른 env·판정) · RobustDexGrasp (미확보) — Stage 1 에서 재검토 |

★ Stage 0 의 teacher 수치는 **폐기하지 않는다.** "반응 없는 정책의 수동 안정성" 이라는
**해석을 붙여서** 하한선 대조군으로 보존한다 (`RESULTS_STAGE0.md`).

## 5. 외란은 시점으로 나눠야 한다 (Stage 0 이 놓친 축)

Stage 0 은 **파지 후**에만 외란을 걸었다. 그런데 open-loop 정책에서 파지 후 외란은
반응할 방법이 없으므로 수동 안정성만 잰다. **시점을 나누면 물어지는 것이 달라진다:**

| 시점 | 정의 (env 가 이미 계산하는 값으로) | 무엇을 묻는가 |
|---|---|---|
| **파지 전** (approach) | `hand_approach_flag == False` (손이 아직 물체 근처가 아님) | t=0 에 세운 **계획이 무효화**되는가 — one-shot 정책의 구조적 취약점 |
| **파지 중** (formation) | `flag == True` AND `delta_z <= 0.1` (손은 닿았고 아직 안 들림) | **접촉 형성**이 깨지는가 — 실패의 다수가 여기 있다 |
| **파지 후** (post-lift) | `delta_z > 0.1` | **유지**가 깨지는가 (Stage 0 이 잰 것) |

★ 시점 정의에 **성공 판정식이 쓰는 값(`flag`, `delta_z`)을 그대로** 쓴다.
별도 기준을 만들면 "언제부터 파지인가"의 정의 불일치가 생긴다 (이전 프로젝트의 게이트 버그 재발 방지).
