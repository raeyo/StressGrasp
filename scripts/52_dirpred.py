#!/usr/bin/env python3
"""방향별 외력 응답 예측기 — 학습 · 사전등록 게이트 판정. 설계 = docs/EXP_DIRPRED.md.

usage: python3 scripts/52_dirpred.py --tag <tag> --feat first|K [--variants P,PT,T,Pocc,PTocc] [--ens 5] npz...
★ 게이트 임계값은 EXP_DIRPRED.md §4 에 수치 보기 전에 고정돼 있다. 여기서 바꾸지 않는다.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device(os.environ.get("GS_CRIT_DEV", "cuda:0" if torch.cuda.is_available() else "cpu"))
HOLD = lambda o: (o % 4) == 3
NPTS = 128
DIRS = ["+x", "-x", "+y", "-y", "+z", "-z"]
# 게이트 (EXP_DIRPRED §4) — 고정
G_A1_AUC, G_A1_FRAC = 0.65, 2.0 / 3.0
G_A2_RATIO, G_A3_RATIO = 1.3, 1.2
G_B0_CONTACT = 0.7
G_B1_GAIN, G_B1_GAIN_OCC = 0.05, 0.10
DEGEN_LO, DEGEN_HI = 0.05, 0.95


def load(paths):
    D, off = [], 0
    for p in paths:
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        d["scene"] = d["scene"] + off
        off = int(d["scene"].max()) + 1
        D.append(d)
    out = {}
    for k in D[0]:
        if k.startswith("meta_"):
            out[k] = int(D[0][k])
        else:
            out[k] = np.concatenate([d[k] for d in D], 0)
    reps = [r for r in "ABCDE" if ("dir_" + r) in out]
    return out, reps


def cf_slice(d):
    nh = d.get("meta_nhand", 6); nft = d.get("meta_nft", 5); ncf = d.get("meta_ncf", 6)
    s = nh + 7 + 3 * nft
    return s, s + 3 * ncf, ncf


def auc(score, label):
    """Mann-Whitney AUC (동순위 평균). label 은 0/1."""
    pos = label > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def occlude(P, rng):
    """반공간 절단 가림 (EXP_DIRPRED §3). P [N,k,3] -> 같은 모양 (남은 점 복제 패딩)."""
    N, k, _ = P.shape
    v = rng.normal(size=(N, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    c = P.mean(1, keepdims=True)
    keep = np.einsum("nkd,nd->nk", P - c, v) <= 0
    out = np.empty_like(P)
    for i in range(N):
        idx = np.where(keep[i])[0]
        if len(idx) < 8:
            idx = np.arange(k)
        out[i] = P[i, rng.choice(idx, k, replace=True)]
    return out


class PointNet(nn.Module):
    def __init__(self, dout=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 64), nn.ELU(), nn.Linear(64, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU())
        self.g = nn.Sequential(nn.Linear(128, dout), nn.ELU())

    def forward(self, p):
        return self.g(self.f(p).max(1).values)


class Net(nn.Module):
    def __init__(self, dvec, use_pcl):
        super().__init__()
        self.use_pcl = use_pcl
        self.pn = PointNet(128) if use_pcl else None
        self.tv = nn.Sequential(nn.Linear(dvec, 64), nn.ELU()) if dvec > 0 else None
        din = (128 if use_pcl else 0) + (64 if dvec > 0 else 0)
        self.h = nn.Sequential(nn.Linear(din, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 6))

    def forward(self, v, p):
        z = []
        if self.use_pcl:
            z.append(self.pn(p))
        if self.tv is not None:
            z.append(self.tv(v))
        return self.h(torch.cat(z, -1))


def fit_one(seed, Xtr, Ptr, ytr, Xva, Pva, yva, epochs=60, bs=1024):
    torch.manual_seed(seed)
    dvec = Xtr.shape[1] if Xtr is not None else 0
    net = Net(dvec, Ptr is not None).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, bad = 1e9, None, 0
    n = len(ytr)
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            loss = bce(net(Xtr[b] if dvec else None, Ptr[b] if Ptr is not None else None), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(bce(net(Xva if dvec else None, Pva), yva))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(best_state); net.eval()
    return net, best


def predict(nets, X, P):
    with torch.no_grad():
        ps = torch.stack([torch.sigmoid(nt(X, P)) for nt in nets], 0)   # [E,N,6]
    return ps.mean(0).cpu().numpy(), ps.std(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat", default="first", choices=["first", "K"])
    ap.add_argument("--variants", default="P,PT,T,Pocc,PTocc")
    ap.add_argument("--ens", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    t0 = time.time()
    d, reps = load(a.paths)
    N0 = len(d["obj"])
    ok = np.ones(N0, bool)
    for r in reps:
        ok &= d["nwin_" + r] > 0
    if a.feat == "K":
        okK = d["capK"] > 0
        capK_rate = float(okK[ok].mean())
        ok &= okK
    else:
        capK_rate = float("nan")
    excl = 1.0 - ok.mean()
    d = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    N = len(d["obj"])
    Y = np.mean(np.stack([d["dir_" + r] for r in reps], 0), 0).astype(np.float32)     # [N,6] 버팀 확률
    C = Y.sum(1)
    objs = d["obj"].astype(np.int64)
    te = HOLD(objs); tr_all = ~te
    rng = np.random.RandomState(0)
    va_sc = np.unique(d["scene"][tr_all]); rng.shuffle(va_sc)
    va_set = set(va_sc[:max(1, len(va_sc) // 5)].tolist())
    va = tr_all & np.array([s in va_set for s in d["scene"]]); tr = tr_all & ~va

    obs = d["obs" if a.feat == "first" else "obsK"].astype(np.float32)
    pcl = d["pcl" if a.feat == "first" else "pclK"].astype(np.float32)
    pcl = pcl[:, ::max(1, pcl.shape[1] // NPTS)][:, :NPTS] / 0.1
    s0, s1, ncf = cf_slice(d)
    cf = obs[:, s0:s1]                                            # priv_obs 에서 ×0.1 된 힘 (palm 좌표계)
    cf_n = np.linalg.norm(cf.reshape(N, ncf, 3), axis=-1) / 0.1   # N 단위
    contact = (cf_n > 0.05).any(1)
    contact_rate = float(contact.mean())
    cf_sum = float(cf_n.sum(1).mean())
    mu, sd = cf[tr].mean(0), cf[tr].std(0) + 1e-6
    cfn = (cf - mu) / sd
    # (탐색적, 사전등록 외) 손 상태 = 정규화 관절 hq + 손바닥 좌표계 손끝 위치 ft_rel
    nh = d.get("meta_nhand", 6); nft = d.get("meta_nft", 5)
    hand = np.concatenate([obs[:, :nh], obs[:, nh + 7:nh + 7 + 3 * nft]], 1)
    hmu, hsd = hand[tr].mean(0), hand[tr].std(0) + 1e-6
    handn = (hand - hmu) / hsd
    pcl_occ = occlude(pcl, np.random.RandomState(1))
    # 가림 학습용: train 의 50% 를 가림으로 교체 (고정 시드)
    mix = np.random.RandomState(2).rand(N) < 0.5
    pcl_mix = np.where(mix[:, None, None], pcl_occ, pcl)

    # 퇴화 축 (홀드아웃 양성 비율)
    pos_rate = (Y[te] > 0.5).mean(0)
    nondeg = (pos_rate >= DEGEN_LO) & (pos_rate <= DEGEN_HI)
    print("== %s feat=%s == 후보 %d (원 %d, 제외 %.1f%%, capK율 %s) 반복 %d 장면 %d 물체 %d | train %d val %d holdout %d"
          % (a.tag, a.feat, N, N0, 100 * excl, "%.3f" % capK_rate if capK_rate == capK_rate else "-",
             len(reps), len(np.unique(d["scene"])), len(np.unique(objs)), tr.sum(), va.sum(), te.sum()))
    print("   접촉률(포착 시점, >0.05N 센서≥1) = %.3f | 6센서 힘 합 평균 = %.3f N | 슬롯0 카운트 = %.3f liftSR = %.3f"
          % (contact_rate, cf_sum, C[d["slot"] == 0].mean(), np.mean([d["succ_" + r][d["slot"] == 0].mean() for r in reps])))
    print("   홀드아웃 양성 비율: " + " ".join("%s=%.3f%s" % (DIRS[i], pos_rate[i], "" if nondeg[i] else "(퇴화)") for i in range(6)))

    T = lambda x: torch.tensor(x, device=DEV)
    yt = T(Y)
    VAR = {
        "P": (None, pcl, pcl), "PT": (cfn, pcl, pcl), "T": (cfn, None, None),
        "Pocc": (None, pcl_mix, pcl_occ), "PTocc": (cfn, pcl_mix, pcl_occ),
        "PH": (handn, pcl, pcl), "PHT": (np.concatenate([handn, cfn], 1), pcl, pcl), "H": (handn, None, None),
    }
    want = [v for v in a.variants.split(",") if v in VAR]
    res = dict(tag=a.tag, feat=a.feat, n=int(N), excl=float(excl), capK_rate=capK_rate,
               contact_rate=contact_rate, cf_sum=cf_sum,
               slot0_cnt=float(C[d["slot"] == 0].mean()), pos_rate=pos_rate.tolist(), nondeg=nondeg.tolist(),
               holdout_objs=sorted(set(objs[te].tolist())), var={})
    PRED = {}
    for name in want:
        X, Ptr_, Pte_ = VAR[name]
        Xt = T(X) if X is not None else None
        Ptr_t = T(Ptr_) if Ptr_ is not None else None
        Pte_t = T(Pte_) if Pte_ is not None else None
        nets = []
        for e in range(a.ens):
            nt, vl = fit_one(e, Xt[tr] if X is not None else None, Ptr_t[tr] if Ptr_ is not None else None, yt[tr],
                             Xt[va] if X is not None else None, Ptr_t[va] if Ptr_ is not None else None, yt[va],
                             epochs=a.epochs)
            nets.append(nt)
        # 평가: 비가림 홀드아웃 + (가림 변형이면) 가림 홀드아웃
        pm, ps = predict(nets, Xt[te] if X is not None else None, T(pcl[te]) if Ptr_ is not None else None)
        row = dict(auc=[auc(pm[:, i], Y[te][:, i]) for i in range(6)],
                   rho_cnt=spearman(pm.sum(1), C[te]),
                   std_mean=float(ps.mean()), err_mean=float(np.abs(pm - Y[te]).mean()))
        # A2: 앙상블 std 상위/하위 25% 의 오차
        u = ps.mean(1); err = np.abs(pm - Y[te]).mean(1)
        q1, q3 = np.percentile(u, [25, 75])
        row["a2_ratio"] = float(err[u >= q3].mean() / max(err[u <= q1].mean(), 1e-9))
        # 최약축 식별 (비퇴화 축만, 홀드아웃)
        if nondeg.sum() >= 2:
            idx = np.where(nondeg)[0]
            row["argmin_agree"] = float((idx[pm[:, idx].argmin(1)] == idx[Y[te][:, idx].argmin(1)]).mean())
        # ECE
        pf, yf = pm.ravel(), Y[te].ravel()
        bins = np.clip((pf * 10).astype(int), 0, 9)
        row["ece"] = float(sum(np.abs(pf[bins == b].mean() - yf[bins == b].mean()) * (bins == b).mean()
                               for b in range(10) if (bins == b).any()))
        if Pte_ is not None and name.endswith("occ") or name == "P":
            pmo, pso = predict(nets, Xt[te] if X is not None else None, T(pcl_occ[te]))
            row["auc_occ"] = [auc(pmo[:, i], Y[te][:, i]) for i in range(6)]
            row["std_mean_occ"] = float(pso.mean())
            row["a3_ratio"] = float(pso.mean() / max(ps.mean(), 1e-9))
        PRED[name] = dict(pm=pm, ps=ps)
        if "auc_occ" in row:
            PRED[name].update(pmo=pmo, pso=pso)
        res["var"][name] = row
        nd = np.array(row["auc"])[nondeg]
        print(("  %-6s AUC " % name) + " ".join("%s=%.3f" % (DIRS[i], row["auc"][i]) for i in range(6))
              + " | 비퇴화평균 %.3f | rho(cnt) %.3f | std %.4f | A2비 %.2f | ECE %.3f%s%s  (%.0fs)"
              % (np.nanmean(nd) if len(nd) else float("nan"), row["rho_cnt"], row["std_mean"], row["a2_ratio"], row["ece"],
                 (" | 가림AUC평균 %.3f A3비 %.2f" % (np.nanmean(np.array(row["auc_occ"])[nondeg]), row["a3_ratio"])) if "auc_occ" in row else "",
                 (" | argmin일치 %.3f" % row["argmin_agree"]) if "argmin_agree" in row else "", time.time() - t0), flush=True)

    # ---------------------------------------------------------------- 게이트
    g = {}
    if "P" in res["var"]:
        au = np.array(res["var"]["P"]["auc"])[nondeg]
        g["A1"] = dict(frac=float(np.mean(au >= G_A1_AUC)) if len(au) else float("nan"), n_axes=int(nondeg.sum()))
        g["A1"]["pass"] = bool(len(au) and g["A1"]["frac"] >= G_A1_FRAC - 1e-9)
        g["A2"] = dict(ratio=res["var"]["P"]["a2_ratio"]); g["A2"]["pass"] = bool(g["A2"]["ratio"] >= G_A2_RATIO)
        g["A3"] = dict(ratio=res["var"]["P"]["a3_ratio"]); g["A3"]["pass"] = bool(g["A3"]["ratio"] >= G_A3_RATIO)
    g["B0"] = dict(contact_rate=contact_rate); g["B0"]["pass"] = bool(contact_rate >= G_B0_CONTACT)
    if "P" in res["var"] and "PT" in res["var"]:
        gain = float(np.nanmean((np.array(res["var"]["PT"]["auc"]) - np.array(res["var"]["P"]["auc"]))[nondeg]))
        g["B1"] = dict(gain=gain)
        if "Pocc" in res["var"] and "PTocc" in res["var"]:
            g["B1"]["gain_occ"] = float(np.nanmean((np.array(res["var"]["PTocc"]["auc_occ"]) - np.array(res["var"]["Pocc"]["auc_occ"]))[nondeg]))
        g["B1"]["pass"] = bool(gain >= G_B1_GAIN or g["B1"].get("gain_occ", -9) >= G_B1_GAIN_OCC)
    res["gates"] = g
    print("== 게이트 ==")
    for k, v in g.items():
        print("   %s %s %s" % (k, "PASS" if v["pass"] else "FAIL", {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "pass"}))
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/dirpred_%s_%s.json" % (a.tag, a.feat), "w"), indent=1)
    # 예측값 저장 (탐색적 분석용 — 게이트 판정에는 쓰지 않는다). 홀드아웃만.
    os.makedirs("runs/critic/_dp", exist_ok=True)
    np.savez_compressed("runs/critic/_dp/pred_%s_%s.npz" % (a.tag, a.feat), Y=Y[te], obj=objs[te], scene=d["scene"][te],
                        slot=d["slot"][te], contact=contact[te], nondeg=nondeg,
                        **{"%s_%s" % (nm, k): v for nm, dd in PRED.items() for k, v in dd.items()})
    print("saved results/dirpred_%s_%s.json  (%.0fs)" % (a.tag, a.feat, time.time() - t0))


if __name__ == "__main__":
    main()
