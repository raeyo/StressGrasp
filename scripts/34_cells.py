#!/usr/bin/env python3
"""셀 비교표 — (손 x 물체셋 x 프레임) 별 lift SR vs 마진. 인자 = 'tag:손:물체셋' 목록."""
import sys, numpy as np
DIRN = ["+x", "-x", "+y", "-y", "+z", "-z"]


def cell(tag):
    d = np.load(f"runs/branch/{tag}/branch.npz", allow_pickle=True)
    n, G = int(d["n_obj"]), int(d["G"])
    su, sr, tg, fired = d["success"], d["success_round"], d["t_grip"], d["fired"]
    al, di, mass = d["alpha"], d["dir_idx"], d["mass"]
    alphas = sorted(set(float(a) for a in d["alphas"]))
    mbar = float(mass[:n].mean())
    S = {a: {dd: [] for dd in range(6)} for a in alphas}
    Ms, oks, mains = [], [], []
    for i, r in enumerate(sr):
        if (tg[i] < 0).all():
            continue          # ★ 실행되지 않은 라운드(빈 행)
        main = su[i, :n] > 0.5
        ok = main & (tg[i] >= 0) & fired[i]
        mains.append(main); oks.append(ok)
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
    M = np.concatenate(Ms); ok = np.concatenate(oks); main = np.concatenate(mains)
    a2 = alphas[-1]
    per = [np.concatenate(S[a2][dd])[ok].mean() for dd in range(6)]
    allok = np.ones(len(ok), bool)
    for dd in range(6):
        allok &= np.concatenate(S[a2][dd])
    return dict(frame=(str(d["frame"]) if "frame" in d.files else "world"), mbar=mbar, lift=main.mean(), Mmean=M[ok].mean(),
                MmN=M[ok].mean() * mbar * 9.81 * 1000, all6=allok[ok].mean(),
                per=per, ratio=max(per) / max(min(per), 1e-3), n=int(ok.sum()),
                worst=DIRN[int(np.argmin(per))], best=DIRN[int(np.argmax(per))], a2=a2)


if __name__ == "__main__":
    print("| 셀 | 프레임 | 물체 평균질량 | **lift SR** | **M 평균** | M (mN) | 전방향생존 @α=2 | 최약/최강 | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    for spec in sys.argv[1:]:
        tag, name = spec.split(":", 1)
        c = cell(tag)
        print(f"| {name} | {c['frame']} | {c['mbar']*1000:.0f} g | **{c['lift']:.3f}** | **{c['Mmean']:.3f}** "
              f"| {c['MmN']:.0f} | {c['all6']:.3f} | {c['ratio']:.0f}× ({c['worst']}/{c['best']}) | {c['n']} |")
