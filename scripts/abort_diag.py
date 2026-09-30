#!/usr/bin/env python3
"""★ 사후·탐색적 — ABORT 판정의 두 가지 되짚기. 판정 아님.
 (1) 예측기가 재는 것이 "들릴까"인가 "버틸까"인가  → g0=1 조건부 AUC
 (2) 사전등록 순차정책은 첫 시도도 무작위였다. **teacher 우선 + 실패 시 재시도**가 배치 현실에 가깝다.
"""
import os, sys, json, argparse, importlib
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("52_dirpred")
ap = importlib.import_module("abort_policy") if False else None
a_ = argparse.ArgumentParser(); a_.add_argument("--hand", required=True); a_.add_argument("--data", required=True)
a_.add_argument("--tag", required=True); a = a_.parse_args()
DEV = m.DEV; NPTS = 128
HOLD = lambda o: (o % 4) == 3
LAYOUT = {"shadow": (18, 5, 6), "inspire": (6, 5, 6), "allegro": (16, 4, 5)}
d = np.load(a.data); nh, nft, ncf = LAYOUT[a.hand]
obj, scene, slot = d["obj"].astype(int), d["scene"].astype(int), d["slot"].astype(int)
R = int(d["meta_R"]); nsc = int(scene.max()) + 1; half = nsc // 2
Y = lambda r: ((d["g0_"+r] > 0) & (d["done_"+r] > 0).all(1) & (d["ypose_"+r] > 0.5).all(1)).astype(np.float32)
yA, yB = Y("A"), Y("B")
g0A, g0B = (d["g0_A"] > 0), (d["g0_B"] > 0)
te = HOLD(obj) & (scene >= half); tr_all = (~HOLD(obj)) & (scene < half)
rng = np.random.RandomState(0); vs = np.unique(scene[tr_all]); rng.shuffle(vs)
vset = set(vs[:max(1, len(vs)//5)].tolist())
va = tr_all & np.isin(scene, list(vset)); tr = tr_all & ~va


class N1(nn.Module):
    def __init__(s, dv):
        super().__init__(); s.pn = m.PointNet(128)
        s.tv = nn.Sequential(nn.Linear(dv, 64), nn.ELU())
        s.h = nn.Sequential(nn.Linear(192, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
    def forward(s, v, p): return s.h(torch.cat([s.pn(p), s.tv(v)], -1))[:, 0]


X = d["obsC"].astype(np.float32); P = d["pclC"].astype(np.float32)
st = max(1, P.shape[1]//NPTS); P = P[:, ::st][:, :NPTS]
X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
Xt, Pt = torch.tensor(X, device=DEV), torch.tensor(P, device=DEV)
tri, vai = torch.where(torch.tensor(tr, device=DEV))[0], torch.where(torch.tensor(va, device=DEV))[0]


def fit(seed, y):
    torch.manual_seed(seed); net = N1(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); bce = nn.BCEWithLogitsLoss()
    best, bsd, bad = 1e9, None, 0
    for ep in range(60):
        net.train(); perm = tri[torch.randperm(len(tri), device=DEV)]
        for s in range(0, len(perm), 1024):
            i = perm[s:s+1024]; l = bce(net(Xt[i], Pt[i]), y[i]); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = float(bce(net(Xt[vai], Pt[vai]), y[vai]))
        if vl < best - 1e-4: best, bad, bsd = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(bsd); net.eval(); return net


nets = [fit(s, torch.tensor(yA, device=DEV)) for s in range(3)]
with torch.no_grad():
    p = torch.sigmoid(torch.stack([n(Xt, Pt) for n in nets], 0).mean(0)).cpu().numpy()
# g0 예측기(비교)
ng = [fit(100+s, torch.tensor(g0A.astype(np.float32), device=DEV)) for s in range(3)]
with torch.no_grad():
    pg = torch.sigmoid(torch.stack([n(Xt, Pt) for n in ng], 0).mean(0)).cpu().numpy()
lift = te & g0B                      # 들린 것들만
print("== (1) 예측기가 재는 것 (%s, 판정 홀드아웃 %d) ==" % (a.hand, te.sum()))
print("   R_all_pose AUC 전체        %.3f" % m.auc(p[te], yB[te]))
print("   g0(들릴까) AUC             %.3f   [전용 g0 예측기 %.3f]" % (m.auc(p[te], g0B[te].astype(float)), m.auc(pg[te], g0B[te].astype(float))))
print("   ★ **들린 것들 안에서** 버팀 AUC %.3f  (n=%d, 양성률 %.3f)"
      % (m.auc(p[lift], yB[lift]), lift.sum(), yB[lift].mean()))

# L 시점도 같은 조건부로 (들린 것만 특징이 존재한다)
XL = d["obsL"].astype(np.float32); PL = d["pclL"].astype(np.float32)
PL = PL[:, ::st][:, :NPTS]
XL = (XL - XL[tr].mean(0)) / (XL[tr].std(0) + 1e-6)
XLt, PLt = torch.tensor(XL, device=DEV), torch.tensor(PL, device=DEV)
_Xt, _Pt = Xt, Pt
Xt, Pt = XLt, PLt
netsL = [fit(200+s, torch.tensor(yA, device=DEV)) for s in range(3)]
with torch.no_grad():
    pL = torch.sigmoid(torch.stack([n(XLt, PLt) for n in netsL], 0).mean(0)).cpu().numpy()
Xt, Pt = _Xt, _Pt
print("   ★ L 시점(들기 후) — 들린 것들 안에서 버팀 AUC %.3f  (같은 n=%d)" % (m.auc(pL[lift], yB[lift]), lift.sum()))

# ---- (2) teacher 우선 순차정책
idx = {}
for i in np.where(te)[0]: idx.setdefault((scene[i], obj[i]), {})[slot[i]] = i
units = {k: v for k, v in idx.items() if len(v) == R}
tau = float(np.median(p[tr])); rs = np.random.RandomState(1)
orate = float(np.mean(p[te] < tau))
print("\n== (2) teacher 우선 순차정책 (단위 %d, τ=%.3f, 중단율 %.2f) ==" % (len(units), tau, orate))
for B in (1, 2, 3):
    out = {k: [] for k in ("학습", "무작위(동일비용)", "오라클(A→B)", "무중단")}
    cost = {k: [] for k in out}
    for k, sm in units.items():
        ids = np.array([sm[s] for s in range(R)])
        acc = {kk: 0.0 for kk in out}; cc = {kk: 0.0 for kk in out}
        for _ in range(400):
            alts = rs.permutation(np.arange(1, R))[:B-1]
            order = np.concatenate([[0], alts]).astype(int)      # ★ 첫 시도 = teacher
            for nm, sc in (("학습", lambda i: p[i]), ("무작위(동일비용)", lambda i: 0.0 if rs.rand() < orate else 1.0),
                           ("오라클(A→B)", lambda i: yA[i]), ("무중단", lambda i: 1.0)):
                ch = order[-1]
                for j, s in enumerate(order):
                    if sc(ids[s]) >= tau or j == len(order)-1:
                        ch = s; cc[nm] += j+1; break
                acc[nm] += yB[ids[ch]]
        for nm in out: out[nm].append(acc[nm]/400); cost[nm].append(cc[nm]/400)
    row = "   B=%d |" % B
    for nm in ("무중단", "학습", "무작위(동일비용)", "오라클(A→B)"):
        row += "  %s %.3f(%.2f회)" % (nm, np.mean(out[nm]), np.mean(cost[nm]))
    dif = np.array(out["학습"]) - np.array(out["무작위(동일비용)"]); n = len(dif)
    bs = np.array([dif[rs.randint(0, n, n)].mean() for _ in range(2000)])
    print(row + "  | 학습−무작위 %+.3f [%+.3f,%+.3f]" % (dif.mean(), *np.percentile(bs, [2.5, 97.5])))
