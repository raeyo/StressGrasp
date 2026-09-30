#!/usr/bin/env python3
"""HOLDPRED — 실행 전 예측기(PRE 앙상블)로 12-D 계획을 CEM 최적화 → 물리 재생용 후보 파일. 설계 = EXP_HOLDPRED.md §4.
usage: python3 scripts/56_planopt.py --pre runs/critic/_hp/pre_<tag>.pt --test runs/critic/hp_<hand>_test_s99/data.npz --out runs/critic/_hp/cands_<tag>.npz
출력 npz: p0_obj [S, n_obj, 7] · plan [S, n, dp] (슬롯 0 = teacher, 1 = CEM 최선, 2..7 = 다음 엘리트) · pred [S, n, 7] · dnorm [S, n]
"""
import os, sys, argparse, importlib, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("52_dirpred"); h = importlib.import_module("55_holdpred")
DEV = m.DEV; NPTS = 128
ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pop", type=int, default=64); ap.add_argument("--gens", type=int, default=8)
ap.add_argument("--sigma", type=float, default=0.1); ap.add_argument("--trust", type=float, default=0.3)
ap.add_argument("--g0min", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
st = torch.load(a.pre, map_location=DEV)
nets = []
for sd in st["nets"]:
    nt = h.PreNet(st["dvec"]).to(DEV); nt.load_state_dict(sd); nt.eval(); nets.append(nt)
mu, sd_ = torch.tensor(st["mu"], device=DEV), torch.tensor(st["sd"], device=DEV)
z = np.load(a.test)
n_obj, R = int(z["meta_nobj"]), int(z["meta_R"]); S = int(z["p0_obj"].shape[0]); n = n_obj * R
dp = z["plan"].shape[1]
ppre = z["pre_pcl"].astype(np.float32); ppre = ppre[:, ::max(1, ppre.shape[1] // NPTS)][:, :NPTS] / 0.1
torch.manual_seed(a.seed)
PLAN = np.zeros((S, n, dp), np.float32); PRED = np.zeros((S, n, 7), np.float32); DN = np.zeros((S, n), np.float32)


def score(vec, pc):
    with torch.no_grad():
        out = torch.stack([torch.sigmoid(nt((vec - mu) / sd_, pc)) for nt in nets], 0).mean(0)   # [K,7]
    pmin = out[:, :6].min(1).values
    s = torch.where(out[:, 6] >= a.g0min, pmin, -1.0 + out[:, 6])       # g0 미달 후보는 폐기 영역
    return s, out


for s_ in range(S):
    rows = np.where(z["scene"] == s_)[0]
    for o in range(n_obj):
        r0 = rows[(z["obj"][rows] == o) & (z["slot"][rows] == 0)][0]
        teach = torch.tensor(z["teach"][r0], device=DEV); q = torch.tensor(z["pre_q"][r0], device=DEV)
        pc = torch.tensor(ppre[r0], device=DEV).unsqueeze(0)
        mean, sig = teach.clone(), torch.full((dp,), a.sigma, device=DEV)
        lo, hi = (teach - a.trust).clamp(-1, 1), (teach + a.trust).clamp(-1, 1)
        best_pool, best_sc = [], []
        for g in range(a.gens):
            cand = (mean + sig * torch.randn(a.pop, dp, device=DEV)).clamp(lo, hi)
            cand[0] = teach                                                 # teacher 는 항상 후보에 포함
            vec = torch.cat([q.expand(a.pop, -1), teach.expand(a.pop, -1), cand, cand - teach], 1)
            sc, out = score(vec, pc.expand(a.pop, -1, -1))
            top = torch.argsort(sc, descending=True)[:8]
            mean = cand[top].mean(0); sig = (cand[top].std(0) + 0.02).clamp(max=a.sigma)
            best_pool.append(cand[top]); best_sc.append(sc[top])
        pool = torch.cat(best_pool, 0); psc = torch.cat(best_sc, 0)
        order = torch.argsort(psc, descending=True)
        # 슬롯 1..R-1 = 서로 다른 엘리트 (L2 0.02 이상 떨어진 것만)
        chosen = []
        for i in order.tolist():
            if all(torch.norm(pool[i] - pool[j]) > 0.02 for j in chosen):
                chosen.append(i)
            if len(chosen) >= R - 1: break
        while len(chosen) < R - 1: chosen.append(order[0].item())
        plans = torch.stack([teach] + [pool[i] for i in chosen], 0)          # [R, dp]
        vec = torch.cat([q.expand(R, -1), teach.expand(R, -1), plans, plans - teach], 1)
        _, out = score(vec, pc.expand(R, -1, -1))
        for k in range(R):
            idx = k * n_obj + o
            PLAN[s_, idx] = plans[k].cpu().numpy(); PRED[s_, idx] = out[k].cpu().numpy(); DN[s_, idx] = float(torch.norm(plans[k] - teach))
    print("scene %d/%d  teacher pmin %.3f -> opt pmin %.3f | g0 %.3f -> %.3f | ||d|| %.3f"
          % (s_ + 1, S, PRED[s_, :n_obj, :6].min(1).mean(), PRED[s_, n_obj:2 * n_obj, :6].min(1).mean(),
             PRED[s_, :n_obj, 6].mean(), PRED[s_, n_obj:2 * n_obj, 6].mean(), DN[s_, n_obj:2 * n_obj].mean()), flush=True)
np.savez_compressed(a.out, p0_obj=z["p0_obj"], plan=PLAN, pred=PRED, dnorm=DN, obj=z["obj"][:n], slot=z["slot"][:n])
print("saved", a.out)
