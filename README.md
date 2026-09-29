# graspstress — 파지는 들어올린 다음에도 버티는가

> **한 줄**: 다지 손 파지의 성공 판정은 "그 순간 들려 있었다"로 끝난다. 이 프로젝트는 **판정 시점을
> 뒤로 밀고 하중·외력을 걸었을 때 기존 정책들의 성능이 얼마나 떨어지는지**를 재서, 이것이 실제로
> 풀어야 할 문제인지부터 판정한다.

★ **세션 인계 = [`docs/HANDOVER.md`](docs/HANDOVER.md) 부터 읽어라.**

문제 정의 = [`docs/PROBLEM.md`](docs/PROBLEM.md) (★정본, **방법 없음**, 불변)
방법 정본 = [`docs/METHOD.md`](docs/METHOD.md) (★현행 — 측정된 사실 · 실패 분해 · 로드맵)
평가 프로토콜 = [`docs/PROTOCOL.md`](docs/PROTOCOL.md) (판정 규칙은 측정 전 고정)
모델 대장 = [`docs/BASELINES.md`](docs/BASELINES.md) (공개 vs 재현 구분)

## 현재 단계

**Stage 0(문제 실재 판정)은 끝났다** — 파지는 한 옥탄트만 버티고(60~80배 이방성), lift SR 은
버팀을 예측하지 못한다. 그 뒤 개입을 세 번 시도했고(grip residual → plan residual → critic 재순위)
셋 다 teacher 를 넘지 못했다. 실패 원인은 매번 다르게 특정됐다 ([`docs/METHOD.md`](docs/METHOD.md) §2).

**현재 = 지각 기반 제어 라인 S1 직전.** critic 을 목적함수로 12-D 계획을 직접 최적화한다.
근거: 여유는 **+0.71 카운트로 실재**하는데 장면이 바뀌면 전이되지 않고, 무작위 탐색은 그중 0% 를
찾는다 → **장면마다 다시 풀어야 하고, 풀려면 장면을 봐야 한다.**

## 이 프로젝트가 하지 않는 것

- 성공 판정식 수정 — 판정식은 원본 그대로 두고 **적용 시점과 물리 조건만** 바꾼다.
- 남의 코드(DemoGrasp) 직접 수정 — 런타임 패치로만 얹는다.
- 수치를 본 뒤 판정 임계 변경 — 실험 조건은 바꿔도 되고 임계는 못 바꾼다.

## 환경

| | |
|---|---|
| 머신 | kimm-h200 (**GPU 0·1 만** — 공용 서버) |
| 평가 env | DemoGrasp (IsaacGym Preview 4), `projects/tacdexgrasp/references/DemoGrasp` — ★ 남의 코드, 고치지 않는다 |
| conda env | `demograsp` (py3.8) — `conda activate` 금지, 절대경로 실행 |
| 부팅 | `source scripts/_common.sh` |

## 실행

```bash
cd $VITAC_ROOT/projects/graspstress
# D1 유지시간 sweep 한 셀 (코드 변경 0 — episodeLength 오버라이드만)
GS_GPU=0 bash scripts/10_hold_sweep.sh ckpt/inspire.pt ours_bench/ours_S.yaml 140 42 66
# 셀 목록 순차 실행
bash scripts/11_d1_driver.sh 0 runs/cells_gpu0.txt
# 집계
python3 scripts/20_summarize.py
```

## 배치

- `runs/` = 원시 로그 (gitignore, 레포 밖 취급)
- `results/` = 판독 근거가 되는 요약만 (커밋)
- `scripts/_patch/` = 남의 코드에 런타임으로 얹는 우리 패치 (D2/D3 용)

## critic 라인 (2026-09-29~)

버팀 카운트(0~6)를 **RL 보상이 아니라 라벨**로 쓰는 노선. 사전등록 [`docs/EXP_CRITIC.md`](docs/EXP_CRITIC.md)
· 판독 [`docs/RESULTS_CRITIC.md`](docs/RESULTS_CRITIC.md) · 수치 [`results/critic_concept__kimm.md`](results/critic_concept__kimm.md).

```bash
GS_GPU=0 GS_SEED=42 bash scripts/50_dataset.sh ds_s42 55      # 라벨 생성 (3회 반복 측정)
python3 scripts/51_critic.py runs/critic/ds_s*/data.npz        # critic 학습 + 게이트 판정
```
