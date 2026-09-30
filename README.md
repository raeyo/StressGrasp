# graspstress — 기존 파지 정책이 만드는 grasp 는 안정적인가

> **한 줄**: 다지 손 파지의 평가는 "들어올렸는가"로 끝난다. 그러나 문헌의 **안정성**은
> **모든 방향**에 대한 성질이고 품질은 **가장 약한 방향**이 정한다(Ferrari–Canny ε).
> **lift SR 은 점이고 안정성은 공이다.** 이 프로젝트는 그 결손을 재고, **ε 을 목적으로 삼아 올린다.**

★ **여기서부터 읽어라.** 문서 **11개**가 전부다.
⚠ 이 레포는 **여러 워크스페이스가 동시에** 작업한다 — 규칙 [`CLAUDE.md`](CLAUDE.md) · 배정 [`CLAIMS.md`](CLAIMS.md).
**master push 는 hub 만.** 작업 디렉토리 하나 = 세션 하나 = 브랜치 하나 (별도 worktree).

## 문서 지도

| 묶음 | 문서 | 무엇 |
|---|---|---|
| **문제** | [`docs/PROBLEM.md`](docs/PROBLEM.md) | 문제 정의 v2 · **사용자 결정 D1~D5** · **결정 1(A/B/C)** · RQ · 반증기준 · v1 이력 |
| **측정 정의** | [`docs/Stability_Metrics.md`](docs/Stability_Metrics.md) | ★ **안정성 지표 A~E 정본** — 정의·프로토콜·선행연구 표·참고문헌 |
| **사실** — 다시 써도 안 죽는다 | [`docs/FACTS.md`](docs/FACTS.md) | 우리가 잰 것 (진술·수치·**반증조건**) |
| | [`docs/LITERATURE.md`](docs/LITERATURE.md) | 문헌 (**출처 등급 필수**) · 공백 G1~G8 |
| | [`docs/PITFALLS.md`](docs/PITFALLS.md) | 함정 P0~P15 · **null 체크리스트** |
| | [`docs/ASSETS.md`](docs/ASSETS.md) | 데이터·모델·하네스·명령 · **없는 것** |
| **생각** — 프레임에 딸림 | [`docs/IDEAS.md`](docs/IDEAS.md) | 아이디어 대장 (**죽은 것 + 되살아나는 조건**) |
| | [`docs/QUESTIONS.md`](docs/QUESTIONS.md) | 미결 (**🔴 Q0 = 방법 라인 선택**) |
| **방법·실험** | [`docs/METHOD.md`](docs/METHOD.md) | ε 을 키우는 축 3개 · Related Work 초안 · 로드맵 |
| | [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | 측정 규약 · **라벨 정의** · 판정 규율 · 재현 파라미터 |
| | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | 실험 대장 (**무효 처리분 포함**) · 다음 실험 T1~T6 |

수치 정본 = `results/` (커밋) · 원시 로그 = `runs/` (**gitignore, 레포 밖 — clone 으로 안 따라온다**)

## 현재 위치

**Stage 0(v1 문제)은 `F1` 로 끝났지만 그것은 축의 한계였다.** v1 은 **무작위 수평 1방향**으로 쟀고,
**결정 1**이 손바닥 **6방향 전부**로 다시 재서 **`B 결손 실재`** 를 판정했다 —
Shadow `R_all(α=8 ≈ 2.4 N)` **.714/.671 < .80**, 자세 유지 `R_all(α=4)` **.568/.529**.
**들고 난 뒤도 다방향으로는 약하고, 낙하 없이도 밀린다**(자세 유지가 물체 유지보다 .1~.2 낮다).

형성 단계에서는 파지가 **한 옥탄트만 버티는 반쪽 케이지**(60~80배, 기제 = 엄지 홀로 대립)다.
⚠ 형성 단계의 "lift SR ↔ 마진 순위 역전"은 **2손 근거가 인용 보류**이고 **파지 후에는 재현되지 않는다.**

그 뒤 **개입을 다섯 번** 시도했고 (grip residual → plan residual → 이진 복원 → 물체수준 오라클 →
critic 재순위) **전부 게이트를 못 넘었다.** 실패 원인은 매번 다르게 특정됐다.

> **재는 데는 성공했고, 바꾸는 데는 아직 한 번도 성공하지 못했다.**

## 지금 당장 할 일

⚠ **방법 라인이 두 갈래이고 사용자 결정 대기 중이다** ([`docs/QUESTIONS.md`](docs/QUESTIONS.md) **Q0**):
**(A)** critic 을 목적함수로 탐색 vs **(B)** ε-보상으로 teacher 를 처음부터 학습.

0. 🔴🔴 **선결 — A→B 인과를 잰다** ([`docs/METHOD.md`](docs/METHOD.md) §0, [`docs/IDEAS.md`](docs/IDEAS.md) I-30).
   현 방법은 전부 **형성 단계(A)** 를 겨냥하는데 결정 1 은 문제를 **B** 로 정의했다.
   같은 상태의 **형성 M ↔ 파지 후 `R_all`** 을 짝지어 상관을 본다. **새 학습 없음**,
   **Q4·Q5(Shadow·Allegro 규약 재측정)와 한 런.** 인과가 없으면 아래가 전부 "B 와 무관한 최적화"가 된다.
1. ★★ **T1** ([`docs/QUESTIONS.md`](docs/QUESTIONS.md) **Q1**, **비용 0**) —
   *12-D 안에 ε 이 큰 파지가 있는가.* 두 방법 라인 공통 전제이고 "없다"면 둘 다 죽는다.
2. 🔴 **Q2** — 우리 지표에 선행이 있다 (**D3Grasp `s_disturb`**, `Stability_Metrics` §8 표에 없는 건). 원문 대조
3. 🟡 **Q8** — sim 손 토크한계·PD 게인이 실기 사양인가 (**위생**)

착수 전 `EXP_*.md` 로 **사전등록**하고, [`docs/PITFALLS.md`](docs/PITFALLS.md) **§3 null 체크리스트**를 읽는다.

## 이 프로젝트가 하지 않는 것

- **성공 판정식 수정** — 판정식은 원본 그대로 두고 **적용 시점과 물리 조건만** 바꾼다
- **남의 코드(DemoGrasp) 직접 수정** — 런타임 패치로만 얹는다
- **수치를 본 뒤 판정 임계 변경** — 실험 조건은 바꿔도 되고 임계는 못 바꾼다
- **배율 표기**(`×32`, `ext_g 50`) — 물리량·현실 출처로만 (D4)
- **M 의 예측력 주장** — 순환적이다 ([`docs/FACTS.md`](docs/FACTS.md) D4)

## 환경·실행

| | |
|---|---|
| 머신 | kimm-h200 (**GPU 0·1 만** — 공용 서버) |
| 평가 env | DemoGrasp (IsaacGym Preview 4), `projects/tacdexgrasp/references/DemoGrasp` — ★ 남의 코드, 고치지 않는다 |
| conda env | `demograsp` (py3.8) — `conda activate` 금지, 절대경로 실행 |
| 레포 | `git@github.com:raeyo/StressGrasp.git` — ★ **디렉토리명 `graspstress` ≠ 레포명** |
| 머신 | ★ **결정 1 은 hub(RTX 3080), 나머지는 kimm-h200 — 절대값 교차 비교 금지** |

```bash
cd $VITAC_ROOT/projects/graspstress && source scripts/_common.sh && gs_check
GS_GPU=0 GS_SEED=42 GS_CC=1 bash scripts/50_dataset.sh ds_s42 55   # ★ GS_CC=1 없으면 촉각이 죽는다
$DG_PY scripts/51_critic.py runs/critic/ds_s*/data.npz
```
명령 전체 = [`docs/ASSETS.md`](docs/ASSETS.md) §5.
