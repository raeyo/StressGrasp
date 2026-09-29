# SETUP — 새 머신에서 graspstress 재현 (첫 대상: lecun)

> 기준 = hub 의 우산 워크스페이스, 2026-09-29. kimm-h200 은 이 문서 없이
> `scripts/_common.sh` 단독 부팅으로 계속 돈다 (기존 동작 유지).

## §1 무엇이 어디서 오나

| 재료 | 위치 (대상 머신) | 어떻게 |
|---|---|---|
| 이 레포 (scripts·docs·`third_party/demograsp.patch`·`env/`) | `$VITAC_PROJECTS/graspstress` | git |
| conda env `demograsp` | `$VITAC_ENV` | `env/spec.sh` — py3.8.19 · torch 2.3.0+cu121 · `env/requirements.lock` |
| IsaacGym Preview4 | `$VITAC_EXTERNAL/isaacgym` | §2 rsync → `pip install -e` (spec.sh) |
| IsaacGymEnvs 1.5.1 | `$VITAC_EXTERNAL/IsaacGymEnvs` | 〃 |
| DemoGrasp upstream 1255dc5 | `$VITAC_EXTERNAL/DemoGrasp` (.git 포함) | §2 rsync — ★ clone 재료일 뿐, 직접 쓰지 않는다 |
| **DG 작업 사본 (실제 평가 대상)** | `$GS_ROOT/third_party/DemoGrasp` (gitignore) | spec.sh: 1255dc5 + 패치 + assets·ckpt 심볼릭 |
| 물체 자산 (ours_bench 등) | `$VITAC_SHARED/assets` | §2 sync_assets.sh |
| teacher ckpt 7종 (inspire.pt 등) | `$VITAC_SHARED/ckpt/demograsp` | §2 sync_assets.sh |
| demograsp_repro (robustdex 포팅·student) | `$VITAC_PROJECTS/tacdexgrasp/demograsp_repro` | §2 rsync — 패치가 `DG_REPRO` 로 참조 |
| RobustDexGrasp 원본 | `$VITAC_EXTERNAL/RobustDexGrasp` | §2 rsync (선택 — 18_rdx 전용, 없으면 G4 SKIP) |

## §2 hub 에서 할 일 (심기) — 2026-09-29 lecun 에서 실제로 쓴 명령

```bash
cd "$VITAC_ROOT"                        # hub 의 우산 루트
M=lecun; D=/SSDc/raeyoung_kang/vitacgrasp   # 대상 alias · 대상의 우산 경로 (tools/machines.tsv 3열)

tools/sync_assets.sh $M --go            # external/isaacgym · shared/{assets,ckpt,objectsets}
rsync -a --exclude __pycache__ --exclude '*.egg-info' external/IsaacGymEnvs/ $M:$D/external/IsaacGymEnvs/
rsync -a --exclude __pycache__ external/RobustDexGrasp/ $M:$D/external/RobustDexGrasp/   # 선택 (18_rdx)

# ★ DemoGrasp 은 .git 만 보내고 대상에서 원본 checkout — hub 작업트리는 수정돼 있다(그 수정분 = 이 레포의 패치)
ssh $M "mkdir -p $D/external/DemoGrasp"
rsync -a external/DemoGrasp/.git/ $M:$D/external/DemoGrasp/.git/
ssh $M "cd $D/external/DemoGrasp && git checkout -q -f HEAD && rm -rf assets && \
        ln -s ../../shared/assets assets && ln -s ../../shared/ckpt/demograsp ckpt"

# ★ tacdexgrasp 는 비공개 레포 + .git 5.5G → hub 에서 depth-1 clone 을 만들어 보낸다 (커밋 상태만 간다)
git clone -q --depth 1 --no-local file://$PWD/projects/tacdexgrasp /tmp/tdg
git -C /tmp/tdg remote set-url origin https://github.com/raeyo/TacDexGrasp.git
rsync -a /tmp/tdg/ $M:$D/projects/tacdexgrasp/ && rm -rf /tmp/tdg
```

★ `shared/ckpt/demograsp_student/*.pt` 는 `projects/tacdexgrasp/git_share/` 를 가리키는 심볼릭이다 →
tacdexgrasp 가 도착하기 전에 돈 `sync_assets.sh` 의 md5 검증은 불일치로 뜬다. tacdexgrasp 를 보낸 뒤 다시 대조하면 맞는다.

## §3 대상 머신에서 (부팅 → 구축 → 게이트)

```bash
# 길 A — hub 에서 우산 부트스트랩 (env.sh 생성 + 아래까지 자동)
bash tools/bootstrap.sh lecun graspstress --go

# 길 B — 대상에서 직접
source $VITAC_ROOT/env.sh graspstress
bash env/spec.sh                        # exit 0 = 구축 완료 (멱등)
GS_GPU=<빈 장 번호> bash env/gates.sh    # G1–G5 · 하나라도 FAIL → exit 1
# sim 스모크(G5)를 빼려면 GS_GATE_SIM=0 bash env/gates.sh
```

## §4 ★ 패치 출처 주의 — 수치 비교 전 필독

`third_party/demograsp.patch` 는 **hub 의 `external/DemoGrasp` 작업트리(2026-09-29)** 에서 떴다.
kimm 은 `projects/tacdexgrasp/references/DemoGrasp` 라는 **별도 사본**을 쓰므로, 둘이 같다는
보장이 없다. → 새 머신 수치는 kimm 기존 셀·골든과 대조하기 전에는 같은 기준선으로 쓰지 않는다.
해소법: kimm 에서 `git -C projects/tacdexgrasp/references/DemoGrasp diff 1255dc5` 를 떠서
이 패치와 비교한다.

## §5 아직 안 되는 것

- `14_student`(GR00T) — demograsp env 에 `transformers` 등이 없고, student ckpt(`groot_*`)의
  이관 채널이 미정이다. 나머지는 게이트(G1–G5) 통과 즉시 새 머신에서 돈다.
