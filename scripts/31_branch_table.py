#!/usr/bin/env python3
"""분기 마진 요약 — EXP_BRANCH.md §4(S0 게이트)·§5(S1 판정) 를 그대로 적용한다."""
import sys, glob, os
import numpy as np

def load(paths):
    out = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        out.append((os.path.basename(os.path.dirname(p)), d))
    return out

def survival(d):
    """(alpha, dir) 별 생존 = 그 그림자 env 가 최종 성공했는가. main 대비 paired."""
    n = int(d["n_obj"]); su = d["success"]; sr = d["success_round"]
    g, al, di = d["group"], d["alpha"], d["dir_idx"]
    tg = d["t_grip"]                       # [round, n_obj]  grip 진입 스텝 (-1 = 진입못함)
    rows = []
    for i, r in enumerate(sr):
        if r == 0:   # round 0 은 successes 가 비어있음
            continue
        ok_main = su[i, :n] > 0.5
        gripped = tg[i] >= 0 if tg.size else np.ones(n, bool)
        for gi in range(2, int(d["G"])):
            e = np.arange(n) + gi * n
            rows.append(dict(round=int(r), alpha=float(al[e[0]]), dir=int(di[e[0]]),
                             main=ok_main, shadow=su[i, e] > 0.5, gripped=gripped))
    ctrl = [(int(r), su[i, :n] > 0.5, su[i, n:2*n] > 0.5) for i, r in enumerate(sr) if r != 0]
    return rows, ctrl

if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(os.environ.get("GS_ROOT", "."), "runs/branch/*/branch.npz")))
    paths = [p for p in paths if any(t in p for t in sys.argv[1:])] or paths
    allrows, allctrl = [], []
    for tag, d in load(paths):
        r, c = survival(d); allrows += r; allctrl += c
        print(f"[{tag}] G={int(d['G'])} alphas={d['alphas']} dose={int(d['dose'])} offset={d['offset']}")

    # ── S0: 복제 충실도
    m = np.array([x[1] for x in allctrl]); cc = np.array([x[2] for x in allctrl])
    print(f"\n=== S0 복제 충실도 ===\n  main SR={m.mean():.4f}  control SR={cc.mean():.4f}"
          f"  per-object 불일치={np.mean(m != cc):.2%}  (이것이 분기 하니스의 잡음 바닥)")

    # ── S1: 생존 곡선
    print("\n=== S1 생존 곡선 — main 이 성공한 (물체,라운드) 만 (n=paired) ===")
    print(f"{'alpha(g)':>9} {'힘(N,평균)':>11} | " + " ".join(f"{['+x','-x','+y','-y','+z','-z'][i]:>6}" for i in range(6)) + " | {:>7} {:>6}".format("6방향전부", "n"))
    mass = 0.0309
    alphas = sorted(set(r["alpha"] for r in allrows))
    surv_by_alpha = {}
    for a in alphas:
        per = {}
        for dd in range(6):
            sel = [r for r in allrows if r["alpha"] == a and r["dir"] == dd]
            if not sel: continue
            mm = np.concatenate([r["main"] & r["gripped"] for r in sel])
            ss = np.concatenate([r["shadow"] for r in sel])
            per[dd] = ss[mm].mean()
        # 6방향 전부 생존 (같은 물체·라운드에서 AND)
        byrd = {}
        for r in allrows:
            if r["alpha"] != a: continue
            key = r["round"]
            byrd.setdefault(key, {})[r["dir"]] = r
        allok, tot = 0, 0
        for key, dm in byrd.items():
            if len(dm) < 6: continue
            any_r = next(iter(dm.values()))
            mm = any_r["main"] & any_r["gripped"]
            ok = np.ones(len(mm), bool)
            for dd in range(6): ok &= dm[dd]["shadow"]
            allok += int(ok[mm].sum()); tot += int(mm.sum())
        surv_by_alpha[a] = (per, allok / max(tot, 1), tot)
        print(f"{a:>9.2f} {a*mass*9.81:>11.3f} | " + " ".join(f"{per.get(i,float('nan')):>6.3f}" for i in range(6))
              + f" | {allok/max(tot,1):>7.3f} {tot:>6}")
    print("\n  ※ 힘(N) 은 물체 평균질량 30.9g 기준. 생존 = DemoGrasp 원 성공판정 그대로.")
