# graspstress — 파지는 들어올린 다음에도 버티는가

> **한 줄**: 다지 손 파지의 성공 판정은 "그 순간 들려 있었다"로 끝난다. 이 프로젝트는 **판정 시점을
> 뒤로 밀고 하중·외력을 걸었을 때 기존 정책들의 성능이 얼마나 떨어지는지**를 재서, 이것이 실제로
> 풀어야 할 문제인지부터 판정한다.

문제 정의 = [`docs/PROBLEM.md`](docs/PROBLEM.md) (★정본, **방법 없음**)
평가 프로토콜 = [`docs/PROTOCOL.md`](docs/PROTOCOL.md) (판정 규칙은 측정 전 고정)
모델 대장 = [`docs/BASELINES.md`](docs/BASELINES.md) (공개 vs 재현 구분)

## 현재 단계

**Stage 0 — 문제 실재 판정.** 새 모델을 학습하지 않는다. 이미 학습된 공개/재현 ckpt 를 가져와
스트레스 축(유지 시간 · 질량 · 외력)에서 낙폭을 잰다. 낙폭이 기준 미만이면 **프로젝트를 닫는다**
(`docs/PROBLEM.md` §5 F1).

## 이 프로젝트가 하지 않는 것

- 새 정책 학습 · 구조 제안 · 촉각 추가 · 시각 열화 — 전부 **Stage 0 범위 밖**이다.
- 성공 판정식 수정 — 판정식은 원본 그대로 두고 **적용 시점과 물리 조건만** 바꾼다.

## 환경

| | |
|---|---|
| 머신 | kimm-h200 (원 실험) · lecun (확장 중) — 새 머신 = `SETUP.md` |
| 평가 env | DemoGrasp (IsaacGym Preview 4) — ★ 남의 코드, 고치지 않는다 (사본 경로는 `_common.sh` 가 머신별 결정) |
| conda env | `demograsp` (py3.8.19) — `conda activate` 금지, 절대경로 실행 |
| 부팅 | `source $VITAC_ROOT/env.sh graspstress` (우산) · `source scripts/_common.sh` (kimm) |

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
