#!/usr/bin/env python3
"""M(s) = 6방향 전부 생존하는 최대 α. EXP_BRANCH §5 의 Q1~Q3 를 그대로 판정한다.
한 런 안에서만 (물체,라운드) 상태가 같으므로, 사다리 전체가 한 런에 있어야 한다."""
import sys, os
import numpy as np

DIRN = ["+x", "-x", "+y", "-y", "+z", "-z"]


def margins(path):
    d = np.load(path, allow_pickle=True)
    n, G = int(d["n_obj"]), int(d["G"])
    su, sr, tg = d["success"], d["success_round"], d["t_grip"]
    fired = d["fired"] if "fired" in d.files else np.ones_like(tg, bool)
    glen = d["grip_len"] if "grip_len" in d.files else np.zeros_like(tg)
    al, di = d["alpha"], d["dir_idx"]
    alphas = sorted(set(float(a) for a in d["alphas"]))
    recs = []
    for i, r in enumerate(sr):
        if r == 0:
            continue
        main = su[i, :n] > 0.5
        ctrl = su[i, n:2 * n] > 0.5
        grip = tg[i] >= 0
        surv = {}                                   # (alpha,dir) -> bool[n]
        for gi in range(2, G):
            e = np.arange(n) + gi * n
            surv[(float(al[e[0]]), int(di[e[0]]))] = su[i, e] > 0.5
        M = np.zeros(n)                             # 0 = 최저 사다리도 못 버팀
        for a in alphas:
            ok = np.ones(n, bool)
            for dd in range(6):
                ok &= surv[(a, dd)]
            M = np.where(ok, a, M)
        recs.append(dict(round=int(r), offset=max(0, int(r) - 1), main=main, ctrl=ctrl, grip=grip, M=M,
                         surv=surv, t_grip=tg[i], fired=fired[i], grip_len=glen[i]))
    return d, alphas, recs


def report(path, alphas, recs, label):
    n = len(recs[0]["main"])
    main = np.concatenate([r["main"] for r in recs])
    ctrl = np.concatenate([r["ctrl"] for r in recs])
    grip = np.concatenate([r["grip"] for r in recs])
    M = np.concatenate([r["M"] for r in recs])
    off = np.concatenate([np.full(n, r["offset"]) for r in recs])
    fired = np.concatenate([r["fired"] for r in recs])
    glen = np.concatenate([r["grip_len"] for r in recs])
    mass = 0.0309
    # ★ 외력이 실제로 걸린 상태만 센다. 안 걸린 것을 "생존"으로 세면 마진이 부풀려진다 (버그 B1).
    ok = main & grip & fired
    print(f"\n########## {label} ##########")
    print(f"  상태 수: 전체 {len(main)} · grip 진입 {int(grip.sum())} · 외력 실발사 {int((grip & fired).sum())}"
          f" · **main 성공&grip&발사 {int(ok.sum())}**")
    lost = int((grip & main & ~fired).sum())
    print(f"  제외: grip 진입 + main 성공인데 외력 미발사 {lost} 개 (창이 열리기 전 lift 완료)"
          f"  |  grip 지속 median {np.median(glen[grip]):.0f} step")
    print(f"  S0 잡음바닥: control 불일치 {np.mean(main != ctrl):.2%}  (main SR {main.mean():.4f} / control SR {ctrl.mean():.4f})")

    print(f"\n  [Q1] lift 성공한 grasp 들 사이의 마진 M 분포  (M = 6방향 전부 버티는 최대 α, 무게배수)")
    mm = M[ok]
    qs = np.percentile(mm, [10, 25, 50, 75, 90])
    print(f"      p10={qs[0]:.3f}  p25={qs[1]:.3f}  median={qs[2]:.3f}  p75={qs[3]:.3f}  p90={qs[4]:.3f}"
          f"   IQR={qs[3]-qs[1]:.3f}")
    print("      분포:", "  ".join(f"M={a:g}:{np.mean(mm==a):.1%}" for a in [0.0] + alphas))
    print(f"      → 힘 단위(30.9g 물체): median {qs[2]*mass*9.81*1000:.0f} mN,  p90 {qs[4]*mass*9.81*1000:.0f} mN")
    step = alphas[1] / alphas[0] if len(alphas) > 1 else 2
    print(f"      판정: IQR({qs[3]-qs[1]:.3f}) vs 사다리 1칸(×{step:g}) → "
          + ("**퍼져 있다** = lift SR 이 안정성 분산을 숨긴다" if qs[3] > qs[1] else "퍼지지 않음 → F(학습 여지 없음)"))

    gf = grip & fired
    print(f"\n  [Q2] M 이 main 의 최종 성공을 예측하는가 (grip&발사 상태 전부, n={int(gf.sum())})")
    y = main[gf].astype(float); x = M[gf]
    pos, neg = x[y > 0.5], x[y < 0.5]
    if len(pos) and len(neg):
        auc = (np.greater.outer(pos, neg).mean() + 0.5 * np.equal.outer(pos, neg).mean())
        print(f"      AUC = {auc:.3f}   (성공 M̄={pos.mean():.3f} vs 실패 M̄={neg.mean():.3f}, n+={len(pos)} n-={len(neg)})")
        print("      판정: " + ("보상 신호로 쓸 자격 있음 (≥0.65)" if auc >= 0.65 else
                              ("경계 (0.60~0.65)" if auc >= 0.60 else "부적격 (<0.60) — 결과와 무관")))
    print(f"\n  [Q3] 방향 이방성 — α 별 방향별 생존율 (main 성공&grip 상태만)")
    print(f"      {'α(g)':>6} {'힘(mN)':>8} | " + " ".join(f"{s:>6}" for s in DIRN) + " | {:>7}".format("전방향"))
    for a in alphas:
        row = []
        for dd in range(6):
            s = np.concatenate([r["surv"][(a, dd)] for r in recs])
            row.append(s[ok].mean())
        allok = np.ones(len(ok), bool)
        for dd in range(6):
            allok &= np.concatenate([r["surv"][(a, dd)] for r in recs])
        print(f"      {a:>6g} {a*mass*9.81*1000:>8.0f} | " + " ".join(f"{v:>6.3f}" for v in row)
              + f" | {allok[ok].mean():>7.3f}")
    worst = min(range(6), key=lambda dd: np.mean(np.concatenate([r["surv"][(alphas[-1], dd)] for r in recs])[ok]))
    best = max(range(6), key=lambda dd: np.mean(np.concatenate([r["surv"][(alphas[-1], dd)] for r in recs])[ok]))
    print(f"      최약축 {DIRN[worst]} vs 최강축 {DIRN[best]} @α={alphas[-1]:g}")

    print(f"\n  [추가] grip 진입 후 경과 스텝(offset) 별 마진 — 파지가 형성되며 단단해지는가")
    print(f"      {'offset':>7} {'n':>5} {'M median':>9} {'M mean':>8}")
    for o in sorted(set(off[ok].tolist())):
        s = ok & (off == o)
        if s.sum() < 5:
            continue
        print(f"      {o:>7} {int(s.sum()):>5} {np.median(M[s]):>9.3f} {M[s].mean():>8.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d, alphas, recs = margins(p)
        report(p, alphas, recs, os.path.basename(os.path.dirname(p)))
