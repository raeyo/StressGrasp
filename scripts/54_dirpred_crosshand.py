#!/usr/bin/env python3
"""DIRPRED 탐색적 — 손 교차 일반화 (★ 사전등록 외, 판정 아님).
점군 단독 P 모델을 손 A 의 전 물체로 학습하고 손 B 의 전 후보로 평가한다 (라벨 정의·좌표계 동일 = palm ±xyz).
usage: python3 scripts/54_dirpred_crosshand.py --train a.npz [a2.npz] --test b.npz [--feat K]
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import importlib
m = importlib.import_module("52_dirpred")

ap = argparse.ArgumentParser()
ap.add_argument("--train", nargs="+", required=True); ap.add_argument("--test", nargs="+", required=True)
ap.add_argument("--feat", default="K"); ap.add_argument("--ens", type=int, default=5)
a = ap.parse_args()

def prep(paths, feat):
    d, reps = m.load(paths)
    ok = np.ones(len(d["obj"]), bool)
    for r in reps: ok &= d["nwin_" + r] > 0
    if feat == "K": ok &= d["capK"] > 0
    d = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    Y = np.mean(np.stack([d["dir_" + r] for r in reps], 0), 0).astype(np.float32)
    pcl = d["pcl" if feat == "first" else "pclK"].astype(np.float32)
    pcl = pcl[:, ::max(1, pcl.shape[1] // m.NPTS)][:, :m.NPTS] / 0.1
    return Y, pcl, d["scene"]

Ytr, Ptr, sc = prep(a.train, a.feat); Yte, Pte, _ = prep(a.test, a.feat)
rng = np.random.RandomState(0); us = np.unique(sc); rng.shuffle(us); vs = set(us[:max(1, len(us) // 5)].tolist())
va = np.array([s in vs for s in sc]); tr = ~va
T = lambda x: torch.tensor(x, device=m.DEV)
nets = [m.fit_one(e, None, T(Ptr[tr]), T(Ytr[tr]), None, T(Ptr[va]), T(Ytr[va]))[0] for e in range(a.ens)]
pm, ps = m.predict(nets, None, T(Pte))
pos = (Yte > 0.5).mean(0); nd = (pos >= m.DEGEN_LO) & (pos <= m.DEGEN_HI)
au = [m.auc(pm[:, i], Yte[:, i]) for i in range(6)]
print("train=%s (n=%d) -> test=%s (n=%d) feat=%s" % ([os.path.basename(os.path.dirname(p)) for p in a.train], len(Ytr),
      [os.path.basename(os.path.dirname(p)) for p in a.test], len(Yte), a.feat))
print("  test 양성비율 " + " ".join("%s=%.3f%s" % (m.DIRS[i], pos[i], "" if nd[i] else "(퇴)") for i in range(6)))
print("  AUC " + " ".join("%s=%.3f" % (m.DIRS[i], au[i]) for i in range(6)) + " | 비퇴화 평균 %.3f | rho(cnt) %.3f"
      % (np.nanmean(np.array(au)[nd]), m.spearman(pm.sum(1), Yte.sum(1))))
