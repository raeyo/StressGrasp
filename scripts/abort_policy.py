#!/usr/bin/env python3
"""ABORT — 중단·재계획 판정. 설계·임계 = docs/EXP_ABORT.md §3 (측정 전 고정).
usage: python3 scripts/abort_policy.py --tag ab_shadow --hand shadow --data runs/postres/ab_shadow/data.npz
★ 스크립트 번호 미배정 (kimm 범위 50-59 소진) — merge 시 hub 가 번호를 준다.
"""
import os, sys, json, argparse, importlib
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("52_dirpred")
DEV = m.DEV; NPTS = 128
HOLD = lambda o: (o % 4) == 3
# ---- 사전등록 임계 (EXP_ABORT §3). 고치지 않는다.
G_A0_CAPC, G_A0_CTRL, G_A0_TOL = 0.80, 0.95, 0.08
G_A1_AUC, G_A2_GAIN, G_A4_TACT = 0.65, 0.10, 0.03
REF = {"shadow": 0.346, "inspire": 0.578, "allegro": 0.302}
LAYOUT = {"shadow": (18, 5, 6), "inspire": (6, 5, 6), "allegro": (16, 4, 5)}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--hand", required=True)
ap.add_argument("--data", required=True); ap.add_argument("--ens", type=int, default=3)
ap.add_argument("--budget", type=int, default=2); ap.add_argument("--trials", type=int, default=400)
a = ap.parse_args()
d = np.load(a.data)
nh, nft, ncf = LAYOUT[a.hand]
din = d["obsC"].shape[1]
assert nh + 7 + 3 * nft + 3 * ncf + 9 == din, (nh, nft, ncf, din)
CF0, CF1 = nh + 7 + 3 * nft, nh + 7 + 3 * nft + 3 * ncf

obj, scene, slot = d["obj"].astype(int), d["scene"].astype(int), d["slot"].astype(int)
R = int(d["meta_R"]); nsc = int(scene.max()) + 1


def allpose(r):
    return ((d["g0_" + r] > 0) & (d["done_" + r] > 0).all(1) & (d["ypose_" + r] > 0.5).all(1)).astype(np.float32)


yA, yB = allpose("A"), allpose("B")
z = slot == 0
res = dict(tag=a.tag, hand=a.hand, R=R, scenes=nsc, n=int(len(obj)))

# ---------------- A0 계측기
ctrl = float(np.mean([d["ctrl_hold_A"][d["g0_A"] > 0].mean(), d["ctrl_hold_B"][d["g0_B"] > 0].mean()]))
A0 = dict(allpose_slot0=float(((yA + yB) / 2)[z].mean()), capC=float(d["capC"].mean()),
          capL=float(d["capL"].mean()), ctrl_hold=ctrl, g0=float(((d["g0_A"] + d["g0_B"]) / 2).mean()))
A0["pass"] = bool(abs(A0["allpose_slot0"] - REF[a.hand]) <= G_A0_TOL and A0["capC"] >= G_A0_CAPC and ctrl >= G_A0_CTRL)
print("== A0 계측기 (%s, 장면 %d, 후보 %d) ==" % (a.hand, nsc, len(obj)))
print("   슬롯0 R_all_pose %.3f (HOLDPRED %.3f ±%.2f) · capC %.3f (>=%.2f) · capL %.3f · ctrlHold %.3f (>=%.2f) · g0 %.3f"
      % (A0["allpose_slot0"], REF[a.hand], G_A0_TOL, A0["capC"], G_A0_CAPC, A0["capL"], ctrl, G_A0_CTRL, A0["g0"]))
print("   A0 %s" % ("PASS" if A0["pass"] else "FAIL"))
res["A0"] = A0

# ---------------- 분할: 사전등록 = 물체 홀드아웃 ∧ 장면 후반 절반
half = nsc // 2
te = HOLD(obj) & (scene >= half)                 # 사전등록 판정 단위
tr_all = (~HOLD(obj)) & (scene < half)
rng = np.random.RandomState(0)
vs = np.unique(scene[tr_all]); rng.shuffle(vs)
vset = set(vs[:max(1, len(vs) // 5)].tolist())
va = tr_all & np.isin(scene, list(vset)); tr = tr_all & ~va
te_obj = HOLD(obj) & (scene < half)              # 병기: 물체만 홀드아웃
te_sc = (~HOLD(obj)) & (scene >= half)           # 병기: 장면만 홀드아웃
print("   분할: train %d · val %d · **판정 홀드아웃(물체∧장면) %d** · 물체만 %d · 장면만 %d"
      % (tr.sum(), va.sum(), te.sum(), te_obj.sum(), te_sc.sum()))


class AbortNet(nn.Module):
    def __init__(self, dvec, use_pcl):
        super().__init__()
        self.pn = m.PointNet(128) if use_pcl else None
        self.tv = nn.Sequential(nn.Linear(dvec, 64), nn.ELU()) if dvec > 0 else None
        di = (128 if use_pcl else 0) + (64 if dvec > 0 else 0)
        self.h = nn.Sequential(nn.Linear(di, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, v, p):
        zz = []
        if self.pn is not None: zz.append(self.pn(p))
        if self.tv is not None: zz.append(self.tv(v))
        return self.h(torch.cat(zz, -1))[:, 0]


def fit(seed, X, P, y, tri, vai, epochs=60, bs=1024):
    torch.manual_seed(seed)
    net = AbortNet(X.shape[1] if X is not None else 0, P is not None).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    best, bs_, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = tri[torch.randperm(len(tri), device=DEV)]
        for s in range(0, len(perm), bs):
            i = perm[s:s + bs]
            l = bce(net(None if X is None else X[i], None if P is None else P[i]), y[i])
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(bce(net(None if X is None else X[vai], None if P is None else P[vai]), y[vai]))
        if vl < best - 1e-4:
            best, bad, bs_ = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(bs_); net.eval()
    return net


def prep(which, drop_tactile=False, use_pcl=True, use_vec=True):
    X = d["obs" + which].astype(np.float32).copy()
    if drop_tactile: X[:, CF0:CF1] = 0.0
    P = d["pcl" + which].astype(np.float32)
    st = max(1, P.shape[1] // NPTS); P = P[:, ::st][:, :NPTS]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd
    Xt = torch.tensor(X, device=DEV) if use_vec else None
    Pt = torch.tensor(P, device=DEV) if use_pcl else None
    return Xt, Pt


yAt = torch.tensor(yA, device=DEV)
tri = torch.where(torch.tensor(tr, device=DEV))[0]
vai = torch.where(torch.tensor(va, device=DEV))[0]


def run_variant(name, which, **kw):
    X, P = prep(which, **kw)
    nets = [fit(s, X, P, yAt, tri, vai) for s in range(a.ens)]
    with torch.no_grad():
        p = torch.sigmoid(torch.stack([n(None if X is None else X, None if P is None else P) for n in nets], 0).mean(0)).cpu().numpy()
    out = dict(auc=float(m.auc(p[te], yB[te])), auc_obj=float(m.auc(p[te_obj], yB[te_obj])),
               auc_scene=float(m.auc(p[te_sc], yB[te_sc])))
    print("   %-22s AUC 판정 %.3f | 물체만 %.3f · 장면만 %.3f" % (name, out["auc"], out["auc_obj"], out["auc_scene"]))
    return p, out


print("\n== A1/A3/A4 예측 (rep A 라벨로 학습 → rep B 로 평가) ==")
pC, vC = run_variant("C 폐합후(들기 전) 전부", "C")
pC_nt, vC_nt = run_variant("C · 촉각 제외", "C", drop_tactile=True)
pC_pc, vC_pc = run_variant("C · 점군만", "C", use_vec=False)
pC_vec, vC_vec = run_variant("C · 상태벡터만", "C", use_pcl=False)
pL, vL = run_variant("L t0(들기 후) 전부", "L")
res["A1"] = dict(**{"pass": bool(vC["auc"] >= G_A1_AUC)}, **vC)
res["A3"] = dict(C=vC["auc"], L=vL["auc"], diff=vL["auc"] - vC["auc"])
res["A4"] = dict(**{"pass": bool(vC["auc"] - vC_nt["auc"] >= G_A4_TACT)},
                 full=vC["auc"], no_tactile=vC_nt["auc"], diff=vC["auc"] - vC_nt["auc"],
                 pcl_only=vC_pc["auc"], vec_only=vC_vec["auc"])
print(">>> A1 (C 시점 예측) = %s  AUC %.3f (>= %.2f)" % ("PASS" if res["A1"]["pass"] else "FAIL", vC["auc"], G_A1_AUC))
print(">>> A3 (시점의 대가) L %.3f − C %.3f = %+.3f" % (vL["auc"], vC["auc"], res["A3"]["diff"]))
print(">>> A4 (촉각 기여) = %s  전부 %.3f − 촉각제외 %.3f = %+.3f (>= %.2f)"
      % ("PASS" if res["A4"]["pass"] else "FAIL", vC["auc"], vC_nt["auc"], res["A4"]["diff"], G_A4_TACT))

# ---------------- A2 순차 정책
tau = float(np.median(pC[tr]))                    # ★ train 분할의 중앙값 = 중단율 50%, 테스트 누설 없음
units = {}
for i in np.where(te)[0]:
    units.setdefault((scene[i], obj[i]), {})[slot[i]] = i
units = {k: v for k, v in units.items() if len(v) == R}
rs = np.random.RandomState(1)


def simulate(score_fn, B, trials):
    """score_fn(idx)->float. 각 단위에서 무작위 순열로 B회까지 시도, score>=tau 면 채택."""
    outc, cost = [], []
    for k, sm in units.items():
        ids = np.array([sm[s] for s in range(R)])
        o = c = 0.0
        for _ in range(trials):
            perm = rs.permutation(R)[:B]
            acc = perm[-1]
            for j, s in enumerate(perm):
                if score_fn(ids[s]) >= tau or j == B - 1:
                    acc = s; c += j + 1; break
            o += yB[ids[acc]]
        outc.append(o / trials); cost.append(c / trials)
    return np.array(outc), np.array(cost)


learned = lambda i: pC[i]
orate = float(np.mean(pC[te] < tau))
rand_fn = lambda i: (0.0 if rs.rand() < orate else 1.0)          # 같은 중단율, 특징 무시
oracle = lambda i: yA[i]                                          # A 로 판단, B 로 평가 (무편향)
print("\n== A2 순차 정책 (판정 단위 %d, 예산 B=%d, τ=%.3f → 홀드아웃 중단율 %.2f) ==" % (len(units), a.budget, tau, orate))
rows = {}
for nm, fn in (("학습 중단", learned), ("무작위 중단(동일비용)", rand_fn), ("오라클 중단(A→B)", oracle)):
    o, c = simulate(fn, a.budget, a.trials)
    rows[nm] = (o, c)
teach = np.array([yB[units[k][0]] for k in units])
first = np.array([np.mean([yB[units[k][s]] for s in range(R)]) for k in units])
dif = rows["학습 중단"][0] - rows["무작위 중단(동일비용)"][0]
n = len(dif); bs = np.array([dif[rs.randint(0, n, n)].mean() for _ in range(4000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print("   teacher 단독(B=1) %.3f · 무작위 1회 %.3f" % (teach.mean(), first.mean()))
for nm in rows:
    print("   %-22s 성공 %.3f · 평균 시도 %.2f" % (nm, rows[nm][0].mean(), rows[nm][1].mean()))
print("   학습 − 무작위(동일비용) = %+.3f  95%%CI [%+.3f, %+.3f]  (>= +%.2f & CI>0)" % (dif.mean(), lo, hi, G_A2_GAIN))
A2pass = bool(dif.mean() >= G_A2_GAIN and lo > 0)
print(">>> A2 (행동 이득) = %s" % ("PASS" if A2pass else "FAIL"))
res["A2"] = dict(**{"pass": A2pass}, budget=a.budget, tau=tau, abort_rate=orate, n_units=len(units),
                 teacher=float(teach.mean()), random1=float(first.mean()),
                 learned=float(rows["학습 중단"][0].mean()), learned_cost=float(rows["학습 중단"][1].mean()),
                 rand_abort=float(rows["무작위 중단(동일비용)"][0].mean()),
                 oracle=float(rows["오라클 중단(A→B)"][0].mean()), diff=float(dif.mean()), ci=[float(lo), float(hi)])
os.makedirs("results", exist_ok=True)
fn = "results/abort_%s__kimm.json" % a.tag
json.dump(res, open(fn, "w"), indent=1, default=float)
print("saved", fn)
