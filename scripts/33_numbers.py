#!/usr/bin/env python3
"""results/branch_margin.md 생성 — 수치 정본. 인자 = branch.npz 경로들."""
import sys
import numpy as np
DIRN = ["+x", "-x", "+y", "-y", "+z", "-z"]; MASS = 0.0309


def load(path):
    d = np.load(path, allow_pickle=True)
    n, G = int(d["n_obj"]), int(d["G"])
    su, sr, tg = d["success"], d["success_round"], d["t_grip"]
    fired = d["fired"] if "fired" in d.files else np.ones_like(tg, bool)
    glen = d["grip_len"] if "grip_len" in d.files else np.zeros_like(tg)
    al, di = d["alpha"], d["dir_idx"]
    alphas = sorted(set(float(a) for a in d["alphas"]))
    S = {a: {dd: [] for dd in range(6)} for a in alphas}
    Ms, oks, mains, ctrls, offs = [], [], [], [], []
    for i, r in enumerate(sr):
        if r == 0:
            continue
        main = su[i, :n] > 0.5; ctrl = su[i, n:2 * n] > 0.5
        ok = main & (tg[i] >= 0) & fired[i]
        mains.append(main); ctrls.append(ctrl); oks.append(ok)
        offs.append(np.full(n, max(0, int(r) - 1)))
        cur = {}
        for gi in range(2, G):
            e = np.arange(n) + gi * n
            cur[(float(al[e[0]]), int(di[e[0]]))] = su[i, e] > 0.5
        M = np.zeros(n)
        for a in alphas:
            o = np.ones(n, bool)
            for dd in range(6):
                o &= cur[(a, dd)]; S[a][dd].append(cur[(a, dd)])
            M = np.where(o, a, M)
        Ms.append(M)
    return dict(M=np.concatenate(Ms), ok=np.concatenate(oks), main=np.concatenate(mains),
                ctrl=np.concatenate(ctrls), off=np.concatenate(offs), S=S, alphas=alphas,
                frame=str(d["frame"]) if "frame" in d.files else "world")


def block(tag, r):
    a_, ok, M = r["alphas"], r["ok"], r["M"]
    print(f"\n## {tag}  (외력 기준 좌표계: **{r['frame']}**)\n")
    print(f"- main SR **{r['main'].mean():.4f}** / control SR **{r['ctrl'].mean():.4f}** · "
          f"per-object 불일치(잡음바닥) **{np.mean(r['main'] != r['ctrl']):.2%}**")
    print(f"- main 성공 & grip & 외력 실발사 상태: **{int(ok.sum())}**\n")
    q = np.percentile(M[ok], [10, 25, 50, 75, 90])
    print("| p10 | p25 | median | p75 | p90 | IQR |")
    print("|---|---|---|---|---|---|")
    print(f"| {q[0]:.3f} | {q[1]:.3f} | **{q[2]:.3f}** | {q[3]:.3f} | {q[4]:.3f} | {q[3]-q[1]:.3f} |")
    print(f"\n힘(30.9 g): p10 **{q[0]*MASS*9.81*1000:.0f} mN** · median **{q[2]*MASS*9.81*1000:.0f} mN** "
          f"· p90 **{q[4]*MASS*9.81*1000:.0f} mN**\n")
    print("계급별: " + " · ".join(f"M={a:g}: {np.mean(M[ok]==a):.1%}" for a in [0.0] + a_) + "\n")
    print("| α (g) | 힘 (mN) | " + " | ".join(DIRN) + " | 6방향 전부 |")
    print("|---|---|" + "---|" * 7)
    for a in a_:
        row = [np.concatenate(r["S"][a][dd])[ok].mean() for dd in range(6)]
        allok = np.ones(len(ok), bool)
        for dd in range(6):
            allok &= np.concatenate(r["S"][a][dd])
        print(f"| {a:g} | {a*MASS*9.81*1000:.0f} | " + " | ".join(f"{v:.3f}" for v in row)
              + f" | **{allok[ok].mean():.3f}** |")


if __name__ == "__main__":
    print("# 분기 마진 M(s) — 수치 정본\n")
    print("> 판독 = `docs/RESULTS_BRANCH.md` · 설계 = `docs/EXP_BRANCH.md` · 원시 = `runs/branch/` (레포 밖)")
    print("> DemoGrasp teacher `inspire.pt` · `ours_S`(33물체) · kimm-h200 GPU 0·1 · dose 1 env-step")
    print("> M(s) = ±xyz 6방향을 **전부** 견디는 최대 α. 생존 = DemoGrasp 원 성공판정 그대로.\n")
    rs = []
    for p in sys.argv[1:]:
        tag = p.split("/")[-2]
        r = load(p); rs.append((tag, r)); block(tag, r)
    if len(rs) >= 2:
        print("\n## seed 반복 폭\n")
        (t1, r1), (t2, r2) = rs[0], rs[1]
        if r1["frame"] == r2["frame"]:
            print(f"- M 평균: {r1['M'][r1['ok']].mean():.4f} vs {r2['M'][r2['ok']].mean():.4f}"
                  f" → **{abs(r1['M'][r1['ok']].mean()-r2['M'][r2['ok']].mean()):.4f}**")
            d = [abs(np.concatenate(r1["S"][1.0][dd])[r1["ok"]].mean()
                     - np.concatenate(r2["S"][1.0][dd])[r2["ok"]].mean()) for dd in range(6)]
            print(f"- 방향별 생존율(α=1) 최대 차: **{max(d):.3f}** ({DIRN[int(np.argmax(d))]})")

