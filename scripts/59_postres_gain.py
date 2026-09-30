#!/usr/bin/env python3
"""POSTRES 판정 — 폐합 후 상수 Δ 의 가동범위. 설계·임계 = docs/EXP_POSTRES.md §4 (측정 전 고정).
usage: python3 scripts/59_postres_gain.py --tag pr_shadow --hand shadow --data runs/postres/pr_shadow/data.npz
지표 = R_all_pose (g0 ∧ 6방향 done ∧ 6방향 Y_pose 전부) — 57_gain.py 와 같은 정의.
"""
import os, json, argparse, numpy as np

# ---- 사전등록 임계 (EXP_POSTRES §4). 수치를 보기 전에 고정됐다. 고치지 않는다.
G_GO, G_COND, G_P2, G_DG0 = 0.10, 0.05, 0.05, -0.05
P0_TOL = 0.08
# HOLDPRED 기준값 (RESULTS_HOLDPRED §1·§2·§3) — 계측기 일치 확인용
REF = {"shadow":  dict(allpose=0.346, g0=0.590, done6=0.402),
       "inspire": dict(allpose=0.578, g0=0.751, done6=0.622),
       "allegro": dict(allpose=0.302, g0=0.529, done6=0.442)}
HOLD = lambda o: (o % 4) == 3

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--hand", required=True)
ap.add_argument("--data", required=True); ap.add_argument("--tagws", default="kimm")
ap.add_argument("--world", action="store_true", help="EXP_WORLDLOAD: 라벨이 바뀌므로 슬롯0 R_all_pose 임계를 걸지 않는다 (§2 W0)")
a = ap.parse_args()
d = np.load(a.data, allow_pickle=True)
names = [str(x) for x in d["meta_armnames"]]
L = d["meta_L"]; K = int(d["meta_K"]); NLEV = L.shape[0]


def allpose(r):
    g0 = d["g0_" + r] > 0
    done = (d["done_" + r] > 0).all(1)
    return (g0 & done & (d["ypose_" + r] > 0.5).all(1)).astype(float), g0.astype(float)


apA, g0A = allpose("A"); apB, g0B = allpose("B")
scene, obj, slot, arm, lev = (d[k].astype(int) for k in ("scene", "obj", "slot", "arm", "lev"))
key = scene * 10000 + obj
units = np.unique(key)
rng = np.random.RandomState(0)

# ---- P0 계측기
z = slot == 0
doneA6 = (d["done_A"] > 0).all(1)
p0 = dict(allpose_slot0=float(((apA + apB) / 2)[z].mean()), g0_slot0=float(((g0A + g0B) / 2)[z].mean()),
          done6=float(doneA6.mean()), ctrl_hold=float(d["ctrl_hold_A"][g0A > 0].mean()),
          on_frac=float(d["on_frac_A"].mean()), n_units=int(len(units)))
ref = REF[a.hand]
p0["pass"] = bool((a.world or abs(p0["allpose_slot0"] - ref["allpose"]) <= P0_TOL)
                  and abs(p0["g0_slot0"] - ref["g0"]) <= P0_TOL
                  and p0["ctrl_hold"] >= 0.95 and p0["done6"] >= 0.30)
print("== P0 계측기 (%s) ==" % a.hand)
print("   슬롯0 R_all_pose %.3f (HOLDPRED %.3f, ±%.2f) · g0 %.3f (%.3f) · ctrlHold %.3f (>=.95) · done6 %.3f (>=.30) · Δ가동 %.2f · 단위 %d"
      % (p0["allpose_slot0"], ref["allpose"], P0_TOL, p0["g0_slot0"], ref["g0"], p0["ctrl_hold"], p0["done6"], p0["on_frac"], p0["n_units"]))
print("   P0 %s" % ("PASS" if p0["pass"] else "FAIL"))

# ---- 팔×레벨별 무편향 best-of-K
idx_of = {}
for i in range(len(key)):
    idx_of.setdefault((key[i], slot[i]), i)
slots_of = {}
for s in np.unique(slot):
    if s == 0: continue
    slots_of.setdefault((int(arm[slot == s][0]), int(lev[slot == s][0])), []).append(int(s))


def boot(v):
    n = len(v)
    if n == 0: return (float("nan"),) * 2
    bs = np.array([v[rng.randint(0, n, n)].mean() for _ in range(4000)])
    return tuple(np.percentile(bs, [2.5, 97.5]))


def cell(ai, li, mask_hold=None):
    ss = slots_of.get((ai, li), [])
    te, bu, bb, rd, gt, gb = [], [], [], [], [], []
    for u in units:
        i0 = idx_of.get((u, 0))
        if i0 is None: continue
        if mask_hold is not None and bool(HOLD(obj[i0])) != mask_hold: continue
        ii = [idx_of[(u, s)] for s in ss if (u, s) in idx_of]
        if not ii: continue
        ii = np.array(ii)
        te.append((apA[i0] + apB[i0]) / 2); gt.append((g0A[i0] + g0B[i0]) / 2)
        j = ii[int(np.argmax(apA[ii]))]                 # A 로 고르고
        bu.append(apB[j]); gb.append((g0A[j] + g0B[j]) / 2)   # B 로 읽는다
        bb.append(float(np.max((apA[ii] + apB[ii]) / 2)))     # 편향 상한
        rd.append(float(np.mean((apA[ii] + apB[ii]) / 2)))
    te, bu, bb, rd = map(np.array, (te, bu, bb, rd))
    dif = bu - te
    lo, hi = boot(dif)
    return dict(n=len(te), teacher=float(te.mean()), rand=float(rd.mean()), best=float(bu.mean()),
                bias=float(bb.mean()), gain=float(dif.mean()), ci=[float(lo), float(hi)],
                dg0=float(np.mean(gb) - np.mean(gt)))


res = dict(tag=a.tag, hand=a.hand, P0=p0, K=K, levels=L.tolist(), cells={}, cells_holdout={})
print("\n== 팔 × 레벨 · 무편향 best-of-%d (전 물체) ==" % K)
print("   %-5s %-3s | %7s %7s %7s | %7s %17s | %7s | %7s" %
      ("팔", "lev", "teach", "rand", "best", "gain", "95%CI", "Δg0", "편향상한"))
for ai in range(1, len(names)):
    for li in range(NLEV):
        c = cell(ai, li)
        res["cells"]["%s_L%d" % (names[ai], li + 1)] = c
        res["cells_holdout"]["%s_L%d" % (names[ai], li + 1)] = cell(ai, li, True)
        print("   %-5s L%-2d | %7.3f %7.3f %7.3f | %+7.3f [%+.3f, %+.3f] | %+7.3f | %7.3f"
              % (names[ai], li + 1, c["teacher"], c["rand"], c["best"], c["gain"], c["ci"][0], c["ci"][1], c["dg0"], c["bias"]))

# ---- P1 / P2 / P3
ok = {k: v for k, v in res["cells"].items() if v["dg0"] >= G_DG0}
bestk = max(ok, key=lambda k: ok[k]["gain"]) if ok else None
star = ok[bestk] if bestk else None
p1 = "NO-GO"
if star and star["ci"][0] > 0:
    p1 = "GO" if star["gain"] >= G_GO else ("조건부GO" if star["gain"] >= G_COND else "NO-GO")
gnew = max([res["cells"][k]["gain"] for k in res["cells"] if k.startswith(("ROT", "NEW"))] or [float("nan")])
gold = max([res["cells"][k]["gain"] for k in res["cells"] if k.startswith("OLD")] or [float("nan")])
p2 = bool(gnew - gold >= G_P2)
p3 = {nm: [round(res["cells"]["%s_L%d" % (nm, i + 1)]["gain"], 3) for i in range(NLEV)] for nm in names[1:]}
res.update(P1=dict(verdict=p1, best_cell=bestk, gain=(star or {}).get("gain"), ci=(star or {}).get("ci"),
                   dg0=(star or {}).get("dg0")),
           P2=dict(**{"pass": p2, "rot_new": gnew, "old": gold, "diff": gnew - gold}),
           P3=p3)
print("\n>>> P1 (가동범위) = **%s**   최선 셀 %s  Δ*=%+.3f CI[%+.3f,%+.3f] Δg0 %+.3f"
      % (p1, bestk, (star or {}).get("gain", float("nan")), *(star or {}).get("ci", [float("nan")] * 2), (star or {}).get("dg0", float("nan"))))
print(">>> P2 (손목 회전이 원인?) = %s   max(ROT,NEW) %+.3f − max(OLD) %+.3f = %+.3f (>= %+.2f)"
      % ("PASS" if p2 else "FAIL", gnew, gold, gnew - gold, G_P2))
print(">>> P3 (클램프 사다리별 이득) = %s" % p3)
rotg = [res["cells"]["ROT_L%d" % (i + 1)]["gain"] for i in range(NLEV)]
w1 = max(rotg) - min(rotg)
res["W1"] = dict(**{"pass": bool(w1 >= 0.15)}, rot_gains=rotg, span=w1)
print(">>> W1 (ROT 레벨 간 이득 범위, 대칭 해소) = %.3f  (>= 0.15 면 손목 방향이 실제 축) %s"
      % (w1, "PASS" if w1 >= 0.15 else "FAIL"))
os.makedirs("results", exist_ok=True)
fn = "results/postres_%s__%s.json" % (a.tag, a.tagws)
json.dump(res, open(fn, "w"), indent=1, default=float)
print("saved", fn)
