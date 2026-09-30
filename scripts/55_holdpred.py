#!/usr/bin/env python3
"""HOLDPRED — hold 상태 예측기(H1) · 실행 전 예측기(H2) 학습 + 게이트 판정. 설계 = docs/EXP_HOLDPRED.md §3·§5.
usage: python3 scripts/55_holdpred.py --tag <tag> [--ens 5] [--save_pre runs/critic/_hp/pre_<tag>.pt] npz...
★ 임계값은 EXP_HOLDPRED.md §5 에 수치 보기 전 고정. 여기서 바꾸지 않는다.
"""
import os, sys, json, argparse, time, importlib
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("52_dirpred")          # load / auc / spearman / PointNet / Net / fit_one / predict
DEV = m.DEV; DIRS = m.DIRS; NPTS = 128
HOLD = lambda o: (o % 4) == 3
G_H0_CTRL, G_H1_AUC, G_H2_AUC, G_H2_G0, G_FRAC = 0.97, 0.65, 0.60, 0.65, 2.0 / 3.0


class PreNet(nn.Module):
    """실행 전 예측기: 월드 점군(물체 중심) + [q, teach, plan, plan-teach] -> 6 방향 로짓 + g0 로짓."""
    def __init__(self, dvec):
        super().__init__()
        self.pn = m.PointNet(128)
        self.tv = nn.Sequential(nn.Linear(dvec, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU())
        self.h = nn.Sequential(nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 7))

    def forward(self, v, p):
        return self.h(torch.cat([self.pn(p), self.tv(v)], -1))


def fit_pre(seed, X, P, Y7, M6, tr, va, epochs=80, bs=1024):
    torch.manual_seed(seed)
    net = PreNet(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_of(idx):
        out = net(X[idx], P[idx])
        l6 = (bce(out[:, :6], Y7[idx, :6]) * M6[idx]).sum() / M6[idx].sum().clamp(min=1)
        lg = bce(out[:, 6], Y7[idx, 6]).mean()
        return l6 + lg
    tri = torch.where(tr)[0]; vai = torch.where(va)[0]
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = tri[torch.randperm(len(tri), device=DEV)]
        for s in range(0, len(perm), bs):
            l = loss_of(perm[s:s + bs]); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_of(vai))
        if vl < best - 1e-4:
            best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(best_state); net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--ens", type=int, default=5)
    ap.add_argument("--save_pre", default=""); ap.add_argument("paths", nargs="+")
    a = ap.parse_args(); t0 = time.time()
    d, _ = m.load(a.paths)
    reps = [r for r in "ABCDE" if ("ypose_" + r) in d]
    N0 = len(d["obj"]); objs = d["obj"].astype(np.int64)
    g0 = np.stack([d["g0_" + r] for r in reps], 1) > 0                  # [N,k]
    done6 = np.stack([(d["done_" + r] > 0).all(1) for r in reps], 1)
    valid = g0.all(1) & done6.all(1)
    Yp = np.mean(np.stack([d["ypose_" + r] for r in reps], 0), 0).astype(np.float32)
    Yh = np.mean(np.stack([d["yhold_" + r] for r in reps], 0), 0).astype(np.float32)
    G0 = g0.mean(1).astype(np.float32)
    ctrl = np.mean(np.stack([d["ctrl_hold_" + r] for r in reps], 0), 0)
    succ0 = np.mean([d["succ_" + r][d["slot"] == 0].mean() for r in reps])
    te = HOLD(objs); tr_all = ~te
    rng = np.random.RandomState(0); vs = np.unique(d["scene"][tr_all]); rng.shuffle(vs)
    vset = set(vs[:max(1, len(vs) // 5)].tolist())
    va = tr_all & np.array([s in vset for s in d["scene"]]); tr = tr_all & ~va
    print("== %s == 후보 %d · g0 %.3f · valid(g0∧done6 양쪽) %.3f · 장면 %d · 물체 %d | train %d val %d holdout %d (valid 기준 %d/%d/%d)"
          % (a.tag, N0, g0.mean(), valid.mean(), len(np.unique(d["scene"])), len(np.unique(objs)),
             tr.sum(), va.sum(), te.sum(), (tr & valid).sum(), (va & valid).sum(), (te & valid).sum()))
    # ---------------------------------------------------------------- H0
    ctrl_ok = float(ctrl[valid].mean())
    A, B = d["ypose_A"][valid], d["ypose_B"][valid]
    cAB = float(np.corrcoef(A.mean(1), B.mean(1))[0, 1])
    cAB_h = float(np.corrcoef(d["yhold_A"][valid].mean(1), d["yhold_B"][valid].mean(1))[0, 1])
    print("   H0: control Y_hold = %.3f (>= %.2f) %s | 슬롯0 lift SR = %.3f · g0 = %.3f | corr(A,B) pose %.3f hold %.3f"
          % (ctrl_ok, G_H0_CTRL, "PASS" if ctrl_ok >= G_H0_CTRL else "FAIL", succ0, g0[d["slot"] == 0].mean(), cAB, cAB_h))
    pos = (Yp[te & valid] > 0.5).mean(0); nondeg = (pos >= m.DEGEN_LO) & (pos <= m.DEGEN_HI)
    posh = (Yh[te & valid] > 0.5).mean(0)
    print("   홀드아웃 양성 비율 Y_pose: " + " ".join("%s=%.3f%s" % (DIRS[i], pos[i], "" if nondeg[i] else "(퇴화)") for i in range(6))
          + " | Y_hold: " + " ".join("%.3f" % x for x in posh) + " | 6방향 전부 pose %.3f" % (Yp[te & valid] > 0.5).all(1).mean())
    res = dict(tag=a.tag, n=int(N0), g0=float(g0.mean()), valid=float(valid.mean()), ctrl_hold=ctrl_ok, slot0_sr=float(succ0),
               corr_ab_pose=cAB, corr_ab_hold=cAB_h, pos_rate=pos.tolist(), nondeg=nondeg.tolist(), var={}, gates={})
    res["gates"]["H0"] = dict(ctrl=ctrl_ok, slot0_sr=float(succ0)); res["gates"]["H0"]["pass"] = bool(ctrl_ok >= G_H0_CTRL)
    T = lambda x: torch.tensor(np.asarray(x, np.float32), device=DEV)
    # ---------------------------------------------------------------- H1: hold 상태
    pcl0 = d["pcl0"].astype(np.float32); pcl0 = pcl0[:, ::max(1, pcl0.shape[1] // NPTS)][:, :NPTS] / 0.1
    nh, nft, ncf = d.get("meta_nhand", 6), d.get("meta_nft", 5), d.get("meta_ncf", 6)
    obs0 = d["obs0"].astype(np.float32); s0 = nh + 7 + 3 * nft
    cf = obs0[:, s0:s0 + 3 * ncf]; hand = np.concatenate([obs0[:, :nh], obs0[:, nh + 7:s0]], 1)
    trv = tr & valid; vav = va & valid; tev = te & valid
    nz = lambda x: (x - x[trv].mean(0)) / (x[trv].std(0) + 1e-6)
    cfn, handn = nz(cf), nz(hand)
    cf_n = np.linalg.norm(cf.reshape(N0, ncf, 3), axis=-1) / 0.1
    print("   t0 접촉률(valid) = %.3f  힘합 %.1f N" % ((cf_n[valid] > 0.05).any(1).mean(), cf_n[valid].sum(1).mean()))
    VAR = {"P": (None, pcl0), "PH": (handn, pcl0), "PHT": (np.concatenate([handn, cfn], 1), pcl0), "H": (handn, None), "T": (cfn, None)}
    yt = T(Yp); yht = T(Yh)
    for name, (X, P) in VAR.items():
        Xt = T(X) if X is not None else None; Pt = T(P) if P is not None else None
        nets = [m.fit_one(e, Xt[trv] if X is not None else None, Pt[trv] if P is not None else None, yt[trv],
                          Xt[vav] if X is not None else None, Pt[vav] if P is not None else None, yt[vav])[0] for e in range(a.ens)]
        pm, ps = m.predict(nets, Xt[tev] if X is not None else None, Pt[tev] if P is not None else None)
        au = [m.auc(pm[:, i], Yp[tev][:, i]) for i in range(6)]
        row = dict(auc=au, auc_nd=float(np.nanmean(np.array(au)[nondeg])) if nondeg.any() else float("nan"),
                   rho_cnt=m.spearman(pm.sum(1), Yp[tev].sum(1)), all6_auc=m.auc(pm.min(1), (Yp[tev] > 0.5).all(1).astype(np.float32)))
        if name == "P":   # Y_hold 병기
            netsh = [m.fit_one(e, None, Pt[trv], yht[trv], None, Pt[vav], yht[vav])[0] for e in range(a.ens)]
            pmh, _ = m.predict(netsh, None, Pt[tev]); row["auc_hold"] = [m.auc(pmh[:, i], Yh[tev][:, i]) for i in range(6)]
        res["var"][name] = row
        print("  %-4s AUC(pose) " % name + " ".join("%s=%.3f" % (DIRS[i], au[i]) for i in range(6))
              + " | 비퇴화평균 %.3f | 6방향전부 AUC %.3f | rho %.3f%s  (%.0fs)"
              % (row["auc_nd"], row["all6_auc"], row["rho_cnt"],
                 (" | Y_hold AUC " + " ".join("%.2f" % x for x in row["auc_hold"])) if "auc_hold" in row else "", time.time() - t0), flush=True)
    auP = np.array(res["var"]["P"]["auc"])[nondeg]
    res["gates"]["H1"] = dict(frac=float(np.mean(auP >= G_H1_AUC)) if len(auP) else float("nan"), n_axes=int(nondeg.sum()))
    res["gates"]["H1"]["pass"] = bool(len(auP) and res["gates"]["H1"]["frac"] >= G_FRAC - 1e-9)
    # ---------------------------------------------------------------- H2: 실행 전
    ppre = d["pre_pcl"].astype(np.float32); ppre = ppre[:, ::max(1, ppre.shape[1] // NPTS)][:, :NPTS] / 0.1
    teach, plan = d["teach"].astype(np.float32), d["plan"].astype(np.float32)
    vec_full = np.concatenate([d["pre_q"].astype(np.float32), teach, plan, plan - teach], 1)
    vec_nop = np.concatenate([d["pre_q"].astype(np.float32), teach], 1)
    Y7 = np.concatenate([Yp, G0[:, None]], 1); M6 = np.repeat(valid[:, None].astype(np.float32), 6, 1)
    Y7t, M6t, Pt = T(Y7), T(M6), T(ppre)
    trt, vat = torch.tensor(tr, device=DEV), torch.tensor(va, device=DEV)
    pre_stats = {}
    for name, vec in (("PRE", vec_full), ("PRE-noplan", vec_nop)):
        mu, sd = vec[tr].mean(0), vec[tr].std(0) + 1e-6
        Xt = T((vec - mu) / sd)
        nets = [fit_pre(e, Xt, Pt, Y7t, M6t, trt, vat) for e in range(a.ens)]
        with torch.no_grad():
            outs = torch.stack([torch.sigmoid(nt(Xt[te], Pt[te])) for nt in nets], 0)
        pm = outs.mean(0).cpu().numpy()
        au = [m.auc(pm[tev[te], i], Yp[tev][:, i]) for i in range(6)]
        aug = m.auc(pm[:, 6], (G0[te] > 0.5).astype(np.float32))
        row = dict(auc=au, auc_nd=float(np.nanmean(np.array(au)[nondeg])) if nondeg.any() else float("nan"), auc_g0=aug,
                   all6_auc=m.auc(pm[tev[te], :6].min(1), (Yp[tev] > 0.5).all(1).astype(np.float32)))
        res["var"][name] = row
        print("  %-10s AUC(pose) " % name + " ".join("%s=%.3f" % (DIRS[i], au[i]) for i in range(6))
              + " | 비퇴화평균 %.3f | 6방향전부 AUC %.3f | g0 AUC %.3f  (%.0fs)" % (row["auc_nd"], row["all6_auc"], aug, time.time() - t0), flush=True)
        if name == "PRE":
            pre_stats = dict(mu=mu, sd=sd, nets=[{k: v.cpu() for k, v in nt.state_dict().items()} for nt in nets], dvec=int(vec.shape[1]))
    auR = np.array(res["var"]["PRE"]["auc"])[nondeg]
    res["gates"]["H2"] = dict(frac=float(np.mean(auR >= G_H2_AUC)) if len(auR) else float("nan"), auc_g0=res["var"]["PRE"]["auc_g0"])
    res["gates"]["H2"]["pass"] = bool(len(auR) and res["gates"]["H2"]["frac"] >= G_FRAC - 1e-9 and res["var"]["PRE"]["auc_g0"] >= G_H2_G0)
    print("== 게이트 ==")
    for k, v in res["gates"].items():
        print("   %s %s %s" % (k, "PASS" if v["pass"] else "FAIL", {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "pass"}))
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/holdpred_%s__kimm.json" % a.tag, "w"), indent=1)
    if a.save_pre:
        torch.save(pre_stats, a.save_pre); print("saved PRE ensemble ->", a.save_pre)
    print("saved results/holdpred_%s__kimm.json (%.0fs)" % (a.tag, time.time() - t0))


if __name__ == "__main__":
    main()
