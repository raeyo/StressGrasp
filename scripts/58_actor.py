#!/usr/bin/env python3
"""ACTOR — PRE critic 앙상블 위에서 amortized actor π(s) 학습 + 재생용 후보 파일. 설계 = docs/EXP_ACTOR.md §1.
usage: python3 scripts/58_actor.py --pre pre.pt --ctx ctx1.npz [ctx2.npz ...] --test test.npz --out cands.npz [--save actor.pt] [--scenes N] [--pess min|mean]
"""
import os, sys, argparse, importlib, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("52_dirpred"); h = importlib.import_module("55_holdpred")
DEV = m.DEV; NPTS = 128
ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True); ap.add_argument("--ctx", nargs="+", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--save", default=""); ap.add_argument("--scenes", type=int, default=0)
ap.add_argument("--pess", default="min"); ap.add_argument("--trust", type=float, default=0.3); ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--noise", default="0.05,0.05,0.1,0.1,0.2,0.2"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed)
st = torch.load(a.pre, map_location=DEV)
nets = []
for sd in st["nets"]:
    nt = h.PreNet(st["dvec"]).to(DEV); nt.load_state_dict(sd); nt.eval()
    for p in nt.parameters(): p.requires_grad_(False)
    nets.append(nt)
mu, sd_ = torch.tensor(st["mu"], device=DEV), torch.tensor(st["sd"], device=DEV)


def ctx_of(paths):
    P, Q, TE = [], [], []
    for pth in paths:
        z = np.load(pth); r0 = np.where(z["slot"] == 0)[0]
        pc = z["pre_pcl"].astype(np.float32); pc = pc[:, ::max(1, pc.shape[1] // NPTS)][:, :NPTS] / 0.1
        P.append(pc[r0]); Q.append(z["pre_q"][r0].astype(np.float32)); TE.append(z["teach"][r0].astype(np.float32))
    return torch.tensor(np.concatenate(P), device=DEV), torch.tensor(np.concatenate(Q), device=DEV), torch.tensor(np.concatenate(TE), device=DEV)


Pc, Qc, Tc = ctx_of(a.ctx); N, dp = Tc.shape
print("actor 학습 컨텍스트 %d (dp=%d) pess=%s trust=%.2f" % (N, dp, a.pess, a.trust))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pn = nets[0].pn                                          # 동결 임베딩
        self.h = nn.Sequential(nn.Linear(128 + 4 + dp, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, dp))
        nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)      # Δ=0 (teacher) 에서 출발

    def forward(self, pc, q, te):
        with torch.no_grad():
            e = self.pn(pc)
        d = self.h(torch.cat([e, q, te], -1))
        return (te + a.trust * torch.tanh(d)).clamp(-1, 1), d


def critic(pc, q, te, plan):
    vec = (torch.cat([q, te, plan, plan - te], -1) - mu) / sd_
    outs = torch.stack([torch.sigmoid(nt(vec, pc)) for nt in nets], 0)    # [K,B,7]
    pmin = outs[:, :, :6].min(2).values; g0 = outs[:, :, 6]
    if a.pess == "min":
        return pmin.min(0).values, g0.min(0).values, outs.mean(0)
    return pmin.mean(0), g0.mean(0), outs.mean(0)


actor = Actor().to(DEV)
opt = torch.optim.Adam(actor.h.parameters(), lr=1e-3)
for ep in range(a.epochs):
    perm = torch.randperm(N, device=DEV)
    tot = 0.0
    for s in range(0, N, 256):
        b = perm[s:s + 256]
        plan, d = actor(Pc[b], Qc[b], Tc[b])
        pmin, g0, _ = critic(Pc[b], Qc[b], Tc[b], plan)
        J = pmin - 2.0 * torch.relu(0.5 - g0) - 0.1 * (d ** 2).mean(1)
        loss = -J.mean(); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    if ep % 50 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            plan, d = actor(Pc, Qc, Tc); pmin, g0, _ = critic(Pc, Qc, Tc, plan); pm_t, g0_t, _ = critic(Pc, Qc, Tc, Tc)
        print("  ep %3d  J=%.4f | pmin(pess) teacher %.3f -> actor %.3f | g0 %.3f -> %.3f | ||Δ|| %.3f"
              % (ep, -tot / N, float(pm_t.mean()), float(pmin.mean()), float(g0_t.mean()), float(g0.mean()), float(torch.norm(plan - Tc, dim=1).mean())), flush=True)
if a.save:
    torch.save({"h": actor.h.state_dict(), "pre": a.pre, "trust": a.trust, "pess": a.pess}, a.save)

# ---------------------------------------------------------------- 후보 파일
z = np.load(a.test)
n_obj, R = int(z["meta_nobj"]), int(z["meta_R"]); S = int(z["p0_obj"].shape[0]); n = n_obj * R
if a.scenes > 0: S = min(S, a.scenes)
pc_all = z["pre_pcl"].astype(np.float32); pc_all = pc_all[:, ::max(1, pc_all.shape[1] // NPTS)][:, :NPTS] / 0.1
noise = [float(x) for x in a.noise.split(",")]
PLAN = np.zeros((S, n, dp), np.float32); PRED = np.zeros((S, n, 7), np.float32); DN = np.zeros((S, n), np.float32)
rng = np.random.RandomState(a.seed)
for s_ in range(S):
    rows = np.where(z["scene"] == s_)[0]
    r0 = np.array([rows[(z["obj"][rows] == o) & (z["slot"][rows] == 0)][0] for o in range(n_obj)])
    pc = torch.tensor(pc_all[r0], device=DEV); q = torch.tensor(z["pre_q"][r0].astype(np.float32), device=DEV); te = torch.tensor(z["teach"][r0].astype(np.float32), device=DEV)
    with torch.no_grad():
        pa, _ = actor(pc, q, te)
        for k in range(R):
            if k == 0: pk = te
            elif k == 1: pk = pa
            else: pk = (pa + noise[(k - 2) % len(noise)] * torch.tensor(rng.randn(n_obj, dp), dtype=torch.float, device=DEV)).clamp(-1, 1)
            _, _, out = critic(pc, q, te, pk)
            PLAN[s_, k * n_obj:(k + 1) * n_obj] = pk.cpu().numpy(); PRED[s_, k * n_obj:(k + 1) * n_obj] = out.cpu().numpy()
            DN[s_, k * n_obj:(k + 1) * n_obj] = torch.norm(pk - te, dim=1).cpu().numpy()
print("test 장면 %d: 예측 pmin teacher %.3f -> actor %.3f | g0 %.3f -> %.3f | ||Δ|| %.3f"
      % (S, PRED[:, :n_obj, :6].min(2).mean(), PRED[:, n_obj:2 * n_obj, :6].min(2).mean(), PRED[:, :n_obj, 6].mean(), PRED[:, n_obj:2 * n_obj, 6].mean(), DN[:, n_obj:2 * n_obj].mean()))
np.savez_compressed(a.out, p0_obj=z["p0_obj"][:S], plan=PLAN, pred=PRED, dnorm=DN, obj=z["obj"][:n], slot=z["slot"][:n])
print("saved", a.out)
