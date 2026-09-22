#!/usr/bin/env python3
"""물체 단위 paired 판정 — PROTOCOL §5.

per_env.npz (runs/stress/*/) 를 읽어 물체별 SR 을 만들고, 기준선(m1/g0) 대비 paired t 를 낸다.
seed-std 로 판정하지 않는다. n = 물체 수 (33/33/34/100).
scipy 없이 동작하도록 t 분포 p 값은 Student-t CDF 를 직접 적분해 구한다.
"""
import json, math, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def per_object_sr(npz):
    d = np.load(npz)
    s, oid, n_obj = d["successes"], d["object_id"], int(d["n_obj"])
    # s: [round, env] -> 물체별 평균 (round x 해당 물체에 배정된 env 전부)
    return np.array([s[:, oid == k].mean() for k in range(n_obj)]), d


def t_sf(t, df):
    """P(T > |t|) * 2 — 정규화 불완전베타를 연분수로 계산 (scipy 없이)."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, it=200):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1e-30 if abs(d) < 1e-30 else d
        d, h = 1.0 / d, 1.0 / d
        for m in range(1, it):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
            c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
            c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d; de = d * c; h *= de
            if abs(de - 1.0) < 3e-12:
                break
        return h

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1 - x) / b
    return max(0.0, min(1.0, ib))


cells = {}
for npz in sorted(ROOT.glob("runs/stress/*/per_env.npz")):
    m = re.match(r"(.+?)__(.+?)__tau(\d+)__m([0-9.]+)__g([0-9.]+)__seed(\d+)$", npz.parent.name)
    if not m:
        continue
    sr, d = per_object_sr(npz)
    cells[npz.parent.name] = dict(
        ckpt=m.group(1), objset=m.group(2), tau=int(m.group(3)),
        mass=float(m.group(4)), ext_g=float(m.group(5)), seed=int(m.group(6)), sr=sr)

out, md = {}, ["# 물체 단위 paired 판정 (PROTOCOL §5)", "",
               "| cell | n | SR | SR(기준) | ΔSR | t | p | 판정 |", "|---|---|---|---|---|---|---|---|"]
for name, c in sorted(cells.items(), key=lambda kv: (kv[1]["objset"], kv[1]["mass"], kv[1]["ext_g"])):
    base = next((b for b in cells.values()
                 if (b["ckpt"], b["objset"], b["seed"], b["tau"]) == (c["ckpt"], c["objset"], c["seed"], c["tau"])
                 and b["mass"] == 1.0 and b["ext_g"] == 0.0), None)
    if base is None or base is c:
        continue
    d = base["sr"] - c["sr"]
    n = len(d)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    p = t_sf(t, n - 1) if math.isfinite(t) else 0.0
    delta = float(d.mean())
    verdict = ("문제 실재" if delta >= 0.10 and p < 0.01 else
               "판정 유보" if delta >= 0.05 else "낙폭 없음")
    out[name] = dict(n=n, sr=float(c["sr"].mean()), sr_base=float(base["sr"].mean()),
                     delta=delta, t=float(t), p=float(p), verdict=verdict)
    md.append(f"| {name} | {n} | {c['sr'].mean():.4f} | {base['sr'].mean():.4f} | "
              f"**{delta:+.4f}** | {t:.2f} | {p:.2e} | {verdict} |")

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "paired.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
(ROOT / "results" / "paired.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
