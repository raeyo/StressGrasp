#!/usr/bin/env python3
"""버팀 카운트 critic — 학습 및 사전등록 게이트 판정. 설계 = docs/EXP_CRITIC.md.

usage: python3 scripts/51_critic.py runs/critic/ds_s42/data.npz runs/critic/ds_s43/data.npz
★ 판정 임계값은 EXP_CRITIC.md §5~§7 에 수치 보기 전에 고정돼 있다. 여기서 바꾸지 않는다.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device(os.environ.get("GS_CRIT_DEV", "cuda:0" if torch.cuda.is_available() else "cpu"))
SEED = 0
HOLD = lambda o: (o % 4) == 3          # 홀드아웃 물체 (EXP_CRITIC §4)
NP_PTS = int(os.environ.get("GS_CRIT_PTS", "128"))


def load(paths):
    D, off = [], 0
    for p in paths:
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        d["scene"] = d["scene"] + off
        off = int(d["scene"].max()) + 1
        D.append(d)
    out = {k: np.concatenate([d[k] for d in D], 0) for k in D[0]}
    reps = [r for r in "ABCDE" if ("cnt_" + r) in out]
    return out, reps


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


class PointNet(nn.Module):
    def __init__(self, dout=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 64), nn.ELU(), nn.Linear(64, 128), nn.ELU())
        self.g = nn.Sequential(nn.Linear(128, dout), nn.ELU())

    def forward(self, p):
        return self.g(self.f(p).max(1).values)


class Critic(nn.Module):
    def __init__(self, dvec, use_pcl):
        super().__init__()
        self.use_pcl = use_pcl
        self.pn = PointNet(128) if use_pcl else None
        din = dvec + (128 if use_pcl else 0)
        self.h = nn.Sequential(nn.Linear(din, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                               nn.Linear(128, 1))

    def forward(self, v, p):
        z = v if not self.use_pcl else torch.cat([v, self.pn(p)], -1)
        return self.h(z).squeeze(-1)


def fit(name, Xtr, Ptr, ytr, Xva, Pva, yva, Xte, Pte, yte, epochs=60, bs=1024):
    torch.manual_seed(SEED)
    use_pcl = Ptr is not None
    net = Critic(Xtr.shape[1], use_pcl).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(Xtr.shape[0], device=DEV)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            loss = ((net(Xtr[b], Ptr[b] if use_pcl else None) - ytr[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(Xva, Pva if use_pcl else None) - yva) ** 2).mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pte = net(Xte, Pte if use_pcl else None).cpu().numpy()
        pva = net(Xva, Pva if use_pcl else None).cpu().numpy()
    rho = spearman(pte, yte.cpu().numpy())
    print("  %-4s val_mse=%.4f  holdout rho=%.3f" % (name, best, rho), flush=True)
    return rho, pte, float(best), spearman(pva, yva.cpu().numpy())


def main(paths):
    d, reps = load(paths)
    N0 = len(d["cnt_A"])
    ok = np.ones(N0, bool)
    for r in reps:
        ok &= d["nwin_" + r] > 0
    excl = 1.0 - ok.mean()
    d = {k: v[ok] for k, v in d.items()}
    N = len(d["cnt_A"])
    C = np.stack([d["cnt_" + r] for r in reps], 1)          # [N, k] 반복 측정
    S = np.stack([d["succ_" + r] for r in reps], 1)
    C1 = np.stack([d["cnt1_" + r] for r in reps], 1)
    k = C.shape[1]
    print("== 데이터 == 후보 %d (원 %d, nwin=0 제외 %.1f%%)  반복 %d  장면 %d  물체 %d"
          % (N, N0, 100 * excl, k, len(np.unique(d["scene"])), len(np.unique(d["obj"]))))
    print("   teacher(slot0) 카운트=%.3f liftSR=%.3f | 전체 카운트=%.3f liftSR=%.3f"
          % (C[d["slot"] == 0].mean(), S[d["slot"] == 0].mean(), C.mean(), S.mean()))
    for s in sorted(set(np.round(d["sig"], 3).tolist())):
        m = np.round(d["sig"], 3) == s
        print("   sigma=%.2f n=%5d 카운트=%.3f liftSR=%.3f" % (s, m.sum(), C[m].mean(), S[m].mean()))

    # ---------------------------------------------------------------- G-A 재현성
    msw = C.var(1, ddof=1).mean()                            # 반복 내 분산
    sig_lab = float(np.sqrt(msw))
    item_mean = C.mean(1)
    msb = float(item_mean.var(ddof=1) * k)                   # one-way ANOVA MSB
    icc = (msb - msw) / (msb + (k - 1) * msw)
    ab = float(np.corrcoef(C[:, 0], C[:, 1])[0, 1])          # 사전등록 형태(2회) ICC
    icc_pre = ab
    gA = (sig_lab <= 1.0) and (icc_pre >= 0.5)
    print("\n== G-A 라벨 재현성 ==")
    print("   sigma_label = %.3f  (기준 <= 1.0)   %s" % (sig_lab, "PASS" if sig_lab <= 1.0 else "FAIL"))
    print("   ICC(사전등록 2회 corr(A,B)) = %.3f  (기준 >= 0.5)   %s" % (icc_pre, "PASS" if icc_pre >= .5 else "FAIL"))
    print("   ICC(1,1) %d회 ANOVA = %.3f | %d회 평균 라벨 신뢰도 = %.3f | 예측 상한 rho <= %.3f"
          % (k, icc, k, k * icc / (1 + (k - 1) * icc), np.sqrt(max(icc, 0))))
    msw1 = C1.var(1, ddof=1).mean()
    ab1 = float(np.corrcoef(C1[:, 0], C1[:, 1])[0, 1])
    print("   [2차 라벨: 첫 window 단독] sigma=%.3f  corr(A,B)=%.3f" % (np.sqrt(msw1), ab1))
    print("   >>> G-A %s" % ("통과" if gA else "불통과 — 컨셉 기각, 이하 재지 않음"))
    res = dict(n=int(N), excl=float(excl), k=int(k), sigma_label=sig_lab, icc_pre=float(icc_pre),
               icc_anova=float(icc), gA=bool(gA),
               teacher_cnt=float(C[d["slot"] == 0].mean()), teacher_sr=float(S[d["slot"] == 0].mean()),
               cnt_all=float(C.mean()), sr_all=float(S.mean()),
               sigma_label_w1=float(np.sqrt(msw1)), icc_pre_w1=float(ab1))
    if not gA:
        json.dump(res, open("results/critic_concept.json", "w"), indent=1)
        return res

    # ---------------------------------------------------------------- G-B 예측
    y = item_mean.astype(np.float32)
    obs = d["obs"].astype(np.float32)
    pcl = d["pcl"].astype(np.float32)[:, ::max(1, d["pcl"].shape[1] // NP_PTS)][:, :NP_PTS] / 0.1
    plan = d["plan"].astype(np.float32)
    objs = d["obj"].astype(np.int64)
    oh = np.zeros((N, int(objs.max()) + 1), np.float32); oh[np.arange(N), objs] = 1
    te = HOLD(objs)
    tr_all = ~te
    va_sc = np.unique(d["scene"][tr_all])
    rng = np.random.RandomState(SEED); rng.shuffle(va_sc)
    va_set = set(va_sc[:max(1, len(va_sc) // 5)].tolist())
    va = tr_all & np.array([s in va_set for s in d["scene"]])
    tr = tr_all & ~va
    mu, sd = obs[tr].mean(0), obs[tr].std(0) + 1e-6
    obsn = (obs - mu) / sd
    T = lambda a: torch.tensor(a, device=DEV)
    print("\n== G-B 예측 == train %d / val %d / holdout %d (물체 %s)"
          % (tr.sum(), va.sum(), te.sum(), sorted(set(objs[te].tolist()))))
    VAR = {
        "B0": (oh, None), "V1": (obsn, None), "V2": (obsn, pcl), "V3": (np.zeros((N, 1), np.float32), pcl),
        "V4": (np.concatenate([obsn, plan], 1), None),
    }
    rho, pred = {}, {}
    yt = T(y)
    for name, (X, P) in VAR.items():
        Xt = T(X); Pt = T(P) if P is not None else None
        r, pte, vm, rv = fit(name, Xt[tr], Pt[tr] if P is not None else None, yt[tr],
                             Xt[va], Pt[va] if P is not None else None, yt[va],
                             Xt[te], Pt[te] if P is not None else None, yt[te])
        rho[name], pred[name] = r, pte
    gB1 = rho["V2"] >= 0.40
    gB2 = (rho["V2"] - rho["V1"]) >= 0.10
    gB3 = (rho["V2"] - rho["B0"]) >= 0.10
    print("   G-B1 rho(V2) = %.3f  (>= 0.40)          %s" % (rho["V2"], "PASS" if gB1 else "FAIL"))
    print("   G-B2 V2-V1   = %.3f  (>= 0.10)          %s" % (rho["V2"] - rho["V1"], "PASS" if gB2 else "FAIL"))
    print("   G-B3 V2-B0   = %.3f  (>= 0.10)          %s" % (rho["V2"] - rho["B0"], "PASS" if gB3 else "FAIL"))
    best_name = max(rho, key=rho.get)
    print("   (최고 변형 = %s, rho=%.3f | rho/sqrt(ICC) = %.3f)"
          % (best_name, rho[best_name], rho[best_name] / max(np.sqrt(max(icc, 1e-9)), 1e-9)))
    res.update(rho={k2: float(v) for k2, v in rho.items()}, gB1=bool(gB1), gB2=bool(gB2), gB3=bool(gB3))
    gB = gB1 and gB2 and gB3
    print("   >>> G-B %s" % ("통과" if gB else "불통과"))

    # ---------------------------------------------------------------- G-C 선택
    # A 의 특징으로 고르고 B 의 측정값으로 읽는다 (선택과 평가 분리)
    pv = pred["V2"]
    sc, ob = d["scene"][te], objs[te]
    cB, cA, sB = C[te, 1], C[te, 0], S[te, 1]
    keys = sc.astype(np.int64) * 1000 + ob
    uk = np.unique(keys)
    rows = []
    for kk in uk:
        m = np.where(keys == kk)[0]
        if len(m) < 2:
            continue
        sl = d["slot"][te][m]
        t_i = m[sl == 0]
        rows.append(dict(teacher=float(cB[t_i[0]]) if len(t_i) else np.nan,
                         rand=float(cB[m].mean()), rand_sr=float(sB[m].mean()),
                         critic=float(cB[m[np.argmax(pv[m])]]),
                         critic_sr=float(sB[m[np.argmax(pv[m])]]),
                         oracle=float(cB[m[np.argmax(cA[m])]])))
    R_ = {k2: np.array([r[k2] for r in rows]) for k2 in rows[0]}
    # 방향별 hold (사전등록 §8 병기 필수)
    DB = d["dir_B"][te]
    DIRSEL = {}
    for kk in uk:
        m = np.where(keys == kk)[0]
        if len(m) < 2:
            continue
        sl = d["slot"][te][m]
        DIRSEL.setdefault("teacher", []).append(DB[m[sl == 0][0]] if (sl == 0).any() else DB[m].mean(0))
        DIRSEL.setdefault("rand", []).append(DB[m].mean(0))
        DIRSEL.setdefault("critic", []).append(DB[m[np.argmax(pv[m])]])
        DIRSEL.setdefault("oracle", []).append(DB[m[np.argmax(cA[m])]])
    print("\n   방향별 hold @303mN (B 측정):  " + "  ".join(["%5s" % x for x in ["+x", "-x", "+y", "-y", "+z", "-z"]]))
    for nm in ("teacher", "rand", "critic", "oracle"):
        v = np.mean(np.stack(DIRSEL[nm]), 0)
        print("   %-8s " % nm + "  ".join("%5.3f" % x for x in v) + "   합=%.3f" % v.sum())
        res.setdefault("dir", {})[nm] = [float(x) for x in v]
    # sigma 수준별 critic 선택 빈도
    sg = np.round(d["sig"][te], 3)
    pick = []
    for kk in uk:
        m = np.where(keys == kk)[0]
        if len(m) >= 2:
            pick.append(sg[m[np.argmax(pv[m])]])
    pick = np.array(pick)
    print("   critic 이 고른 후보의 sigma 분포: " + " ".join("%.2f:%.0f%%" % (s_, 100 * (pick == s_).mean())
                                                        for s_ in sorted(set(sg.tolist()))))
    dif = R_["critic"] - R_["rand"]
    rs = np.random.RandomState(SEED)
    bs = np.array([dif[rs.randint(0, len(dif), len(dif))].mean() for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    gain_or = R_["oracle"] - R_["rand"]
    eta = dif.mean() / gain_or.mean() if gain_or.mean() > 1e-9 else float("nan")
    gC1 = (dif.mean() >= 0.15) and (lo > 0)
    gC2 = (R_["critic_sr"].mean() >= R_["rand_sr"].mean() - 0.05)
    print("\n== G-C 선택 == 홀드아웃 장면 %d개 (장면당 후보 %d)" % (len(rows), int(te.sum() / max(len(rows), 1))))
    for nm in ("teacher", "rand", "critic", "oracle"):
        print("   %-8s 카운트 = %.3f" % (nm, np.nanmean(R_[nm])))
    print("   G-C1 critic-random = %+.3f  95%%CI [%+.3f, %+.3f]  (>= +0.15 & CI>0)   %s"
          % (dif.mean(), lo, hi, "PASS" if gC1 else "FAIL"))
    print("   G-C2 liftSR critic %.3f vs random %.3f (>= -0.05)                    %s"
          % (R_["critic_sr"].mean(), R_["rand_sr"].mean(), "PASS" if gC2 else "FAIL"))
    print("   G-C3 선택 효율 eta = %.3f  (오라클A->B 갭 %+.3f 중)" % (eta, gain_or.mean()))
    res.update(sel={k2: float(np.nanmean(v)) for k2, v in R_.items()},
               gc_diff=float(dif.mean()), gc_ci=[float(lo), float(hi)], eta=float(eta),
               gC1=bool(gC1), gC2=bool(gC2), gB=bool(gB))
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/critic_concept.json", "w"), indent=1)
    print("\n>>> 종합: G-A %s · G-B %s · G-C %s"
          % ("통과" if gA else "불통과", "통과" if gB else "불통과", "통과" if (gC1 and gC2) else "불통과"))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
