#!/usr/bin/env python3
"""HOLDPRED — 행동 이득 판정 H3. 설계 = EXP_HOLDPRED.md §4·§5.
usage: python3 scripts/57_gain.py --tag <tag> --stage1 runs/critic/hp_<hand>_test_s99/data.npz --stage2 runs/critic/hp_<hand>_replay/data.npz [--cands cands.npz]
stage1 = 같은 장면의 teacher(슬롯0) + 무작위 후보(슬롯1..7), A/B 2회 → teacher · random · oracle8
stage2 = 같은 장면(p0 동일)의 teacher(슬롯0) + CEM 최선(슬롯1) + 엘리트(2..7), A/B 2회 → opt
지표 = R_all_pose (6방향 Y_pose 전부 통과; g0=0 이면 0). 단위 = (장면, 물체). 판정은 홀드아웃 물체(obj%4==3).
"""
import os, sys, json, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--stage1", required=True); ap.add_argument("--stage2", required=True)
ap.add_argument("--cands", default=""); a = ap.parse_args()
G_H3_GAIN, G_H3_G0 = 0.10, -0.05
HOLD = lambda o: (o % 4) == 3


def allpose(d, r):
    g0 = d["g0_" + r] > 0; done = (d["done_" + r] > 0).all(1)
    return (g0 & done & (d["ypose_" + r] > 0.5).all(1)).astype(np.float32), g0.astype(np.float32)


def table(d):
    """(scene, obj) -> dict(slot -> (allpose_A, allpose_B, g0_A, g0_B))"""
    aA, gA = allpose(d, "A"); aB, gB = allpose(d, "B")
    out = {}
    for i in range(len(d["obj"])):
        out.setdefault((int(d["scene"][i]), int(d["obj"][i])), {})[int(d["slot"][i])] = (aA[i], aB[i], gA[i], gB[i])
    return out


d1, d2 = np.load(a.stage1), np.load(a.stage2)
T1, T2 = table(d1), table(d2)
cz = np.load(a.cands) if a.cands else None
rows = []
for key in sorted(set(T1) & set(T2)):
    s1, s2 = T1[key], T2[key]
    if 0 not in s1 or 0 not in s2 or 1 not in s2: continue
    rnd = [s1[k] for k in s1 if k > 0]
    ora_pick = max(range(len(rnd)), key=lambda i: rnd[i][0])              # A 로 고르고
    rows.append(dict(scene=key[0], obj=key[1], hold=bool(HOLD(key[1])),
                     teacher1=(s1[0][0] + s1[0][1]) / 2, teacher2=(s2[0][0] + s2[0][1]) / 2,
                     random=float(np.mean([(r[0] + r[1]) / 2 for r in rnd])), oracle8=rnd[ora_pick][1],   # B 로 읽는다
                     opt=(s2[1][0] + s2[1][1]) / 2, opt_best_elite=float(max((s2[k][0] + s2[k][1]) / 2 for k in s2 if k > 0)),
                     g0_teacher=(s1[0][2] + s1[0][3]) / 2, g0_opt=(s2[1][2] + s2[1][3]) / 2))
R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k not in ("hold",)}
H = np.array([r["hold"] for r in rows])
rng = np.random.RandomState(0)


def report(mask, label):
    n = int(mask.sum())
    if n == 0: return {}
    dif = R["opt"][mask] - R["teacher1"][mask]
    bs = np.array([dif[rng.randint(0, n, n)].mean() for _ in range(4000)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    gap = R["oracle8"][mask].mean() - R["teacher1"][mask].mean()
    eta = dif.mean() / gap if abs(gap) > 1e-9 else float("nan")
    dg0 = R["g0_opt"][mask].mean() - R["g0_teacher"][mask].mean()
    out = dict(n=n, teacher1=R["teacher1"][mask].mean(), teacher2=R["teacher2"][mask].mean(), random=R["random"][mask].mean(),
               oracle8=R["oracle8"][mask].mean(), opt=R["opt"][mask].mean(), opt_best_elite=R["opt_best_elite"][mask].mean(),
               diff=dif.mean(), ci=[lo, hi], eta=eta, dg0=dg0, g0_teacher=R["g0_teacher"][mask].mean(), g0_opt=R["g0_opt"][mask].mean())
    print("== %s (단위 %d) ==" % (label, n))
    print("   R_all_pose: teacher(stage1) %.3f · teacher(stage2 재현) %.3f · random %.3f · oracle8(A→B) %.3f · **opt** %.3f · 최선 엘리트(상한, 편향) %.3f"
          % (out["teacher1"], out["teacher2"], out["random"], out["oracle8"], out["opt"], out["opt_best_elite"]))
    print("   opt − teacher = %+.3f  95%%CI [%+.3f, %+.3f] (>= +%.2f & CI>0) | g0 %.3f -> %.3f (Δ %+.3f, >= %.2f) | η = %.2f"
          % (dif.mean(), lo, hi, G_H3_GAIN, out["g0_teacher"], out["g0_opt"], dg0, G_H3_G0, eta))
    return out


res = dict(tag=a.tag, holdout=report(H, "홀드아웃 물체"), all=report(np.ones_like(H), "전 물체 (참고)"))
if res["holdout"]:
    h = res["holdout"]
    ok = (h["diff"] >= G_H3_GAIN) and (h["ci"][0] > 0) and (h["dg0"] >= G_H3_G0)
    res["H3"] = {"pass": bool(ok), "diff": h["diff"], "ci": h["ci"], "dg0": h["dg0"], "eta": h["eta"]}
    print(">>> H3 %s" % ("PASS" if ok else "FAIL"))
if cz is not None:
    # 진단 (a): PRE 예측 pmin vs 실측 (stage2 전 슬롯)
    pred = cz["pred"]; xs, ys = [], []
    for key, s2 in T2.items():
        for k, v in s2.items():
            xs.append(pred[key[0], k * int(cz["obj"].max() + 1) + key[1], :6].min()); ys.append((v[0] + v[1]) / 2)
    xs, ys = np.array(xs), np.array(ys)
    from importlib import import_module; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    m = import_module("52_dirpred")
    print("   진단(a) stage2 전 후보: PRE 예측 pmin vs 실측 R_all_pose  Spearman %.3f · AUC %.3f (n=%d)" % (m.spearman(xs, ys), m.auc(xs, (ys > 0.5).astype(float)), len(xs)))
    res["diag_pre_vs_meas"] = dict(rho=m.spearman(xs, ys), n=int(len(xs)))
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/holdgain_%s__kimm.json" % a.tag, "w"), indent=1, default=float)
print("saved results/holdgain_%s__kimm.json" % a.tag)
