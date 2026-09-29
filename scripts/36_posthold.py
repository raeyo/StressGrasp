#!/usr/bin/env python3
"""posthold 분기 결과 분석 — EXP_POSTLIFT §1 판정규칙(사전등록) 그대로.

usage: python3 scripts/36_posthold.py runs/posthold/<tag>/branch.npz [...] [--out FILE]

- 유효 main 상태 = ph_g0==1 ∧ ph_done(main)==1. 그림자(control·probe)는 done==1 인 것만 통계에 쓴다.
- 물체 유지 Y_hold = ph_ep_end ≤ 5cm ∧ ph_dz_end > 5cm
- 자세 유지 Y_pose = Y_hold ∧ ph_ep_max ≤ 2cm ∧ ph_eR_max ≤ 15°
- R_d / R_mean / R_all = 물체별 평균 먼저 → 물체 macro 평균 (Stability_Metrics §9).
  R_all 은 같은 main 상태에서 분기한 6방향 probe 의 곱(전부 통과). 95% CI = 물체 cluster
  bootstrap 1000회. tag 규약 <hand>_a<alpha>_s<seed> 로 여러 dump 를 (hand, alpha) 로 묶는다.
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np

DIRN = ["+x", "-x", "+y", "-y", "+z", "-z"]
EP_END_MAX, DZ_END_MIN = 0.05, 0.05          # Y_hold (m)
EP_MAX_MAX, ER_MAX_MAX = 0.02, np.deg2rad(15.0)   # Y_pose (m, rad)
GATE_HOLD, GATE_EP_MM = 0.97, 2.0            # EXP_POSTLIFT §2 게이트
NBOOT, BOOTSEED = 1000, 42
RAD2DEG = 180.0 / np.pi


def _k(d, key):
    if key not in d.files:
        raise SystemExit("[36] 키 없음: %s — posthold dump(posthold 모드 branch.npz)가 아니다" % key)
    return d[key]


def macro(vals, objs):
    """물체별 평균 먼저 → 물체 macro 평균 (§9)."""
    vals = np.asarray(vals, dtype=float)
    objs = np.asarray(objs)
    us = np.unique(objs)
    if len(us) == 0:
        return float("nan")
    return float(np.array([vals[objs == u].mean() for u in us]).mean())


def boot_ci(vals, objs, rng):
    """물체 단위 cluster bootstrap 95% CI."""
    vals = np.asarray(vals, dtype=float)
    objs = np.asarray(objs)
    us = np.unique(objs)
    if len(us) < 2:
        return float("nan"), float("nan")
    per = np.array([vals[objs == u].mean() for u in us])
    idx = rng.integers(0, len(us), size=(NBOOT, len(us)))
    means = per[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def parse_tag(path):
    tag = os.path.basename(os.path.dirname(os.path.abspath(path)))
    mt = re.match(r"^(?P<hand>[a-z0-9]+)_a(?P<alpha>[\d.]+)_s(?P<seed>\d+)$", tag)
    if mt is None:
        return tag, None, None, None
    return tag, mt.group("hand"), float(mt.group("alpha")), int(mt.group("seed"))


def load_states(path):
    """한 dump 에서 (유효 main 상태 배열들, 메타) 를 뽑는다."""
    d = np.load(path, allow_pickle=True)
    n, G = int(d["n_main"]), int(d["G"])
    group, main_id, diri = d["group"], d["main_id"], d["dir_idx"]
    alphas = np.atleast_1d(d["alphas"]).astype(float)
    if len(alphas) != 1:
        raise SystemExit("[36] %s: alpha 가 %d 개 — posthold 는 1개 덤프 가정 (35_posthold.sh)" % (path, len(alphas)))
    g0, done = _k(d, "ph_g0"), _k(d, "ph_done")
    ep_end, dz_end = _k(d, "ph_ep_end"), _k(d, "ph_dz_end")
    ep_max, eR_max = _k(d, "ph_ep_max"), _k(d, "ph_eR_max")
    fin = done == 1.0                                   # nan(미시작)·0(미완료) → False
    y_hold = fin & (ep_end <= EP_END_MAX) & (dz_end > DZ_END_MIN)
    y_pose = y_hold & (ep_max <= EP_MAX_MAX) & (eR_max <= ER_MAX_MAX)

    ctrl = np.empty(n, int)
    probe = [np.empty(n, int) for _ in range(6)]
    for e in range(len(group)):
        g = group[e]
        if g == 1:
            ctrl[main_id[e]] = e
        elif g >= 2:
            probe[int(diri[e])][main_id[e]] = e

    valid = (g0 == 1) & (done[:, :n] == 1.0)            # (rounds, n_main)
    r, m = np.nonzero(valid)
    st = dict(
        objs=d["obj_id"][m],
        ctrl_fin=fin[r, ctrl[m]], ctrl_y_hold=y_hold[r, ctrl[m]], ctrl_y_pose=y_pose[r, ctrl[m]],
        ctrl_ep_end=ep_end[r, ctrl[m]],
        p_fin=np.stack([fin[r, probe[dd][m]] for dd in range(6)]),
        p_y_hold=np.stack([y_hold[r, probe[dd][m]] for dd in range(6)]),
        p_y_pose=np.stack([y_pose[r, probe[dd][m]] for dd in range(6)]),
        p_ep_max=np.stack([ep_max[r, probe[dd][m]] for dd in range(6)]),
        p_eR_max=np.stack([eR_max[r, probe[dd][m]] for dd in range(6)]),
    )
    g1 = g0 == 1
    meta = dict(
        path=path, rounds=int(g0.shape[0]), n_states=int(valid.sum()), n_main=int(d["n_main"]),
        alpha=float(alphas[0]),
        G_rate=float(g0.mean()),
        g1_incomplete=float((g1 & ~(done[:, :n] == 1.0)).sum() / max(1, g0.size)),
        g0_zero=float((~g1).mean()),
        sr_main=float(d["success"][:, :n].mean()) if len(d["success"]) else float("nan"),
        sr_rounds=int(d["success"].shape[0]),
        cfg="hold/load/ramp/recover=%d/%d/%d/%d" % (
            int(_k(d, "ph_hold")), int(_k(d, "ph_load")), int(_k(d, "ph_ramp")), int(_k(d, "ph_recover"))),
    )
    return st, meta


def r_all(st, ykey):
    """done==1 probe 들의 곱(전부 통과). done probe 0개 상태는 제외."""
    fin, y = st["p_fin"], st[ykey]
    has = fin.sum(axis=0) > 0
    allpass = (y | ~fin).all(axis=0) & has              # 안 걸린 방향은 곱에서 빠진다
    return allpass, int((~has).sum())


def med_wfinite(v, fin, scale=1.0):
    sel = np.asarray(fin, bool) & ~np.isnan(np.asarray(v, float))
    if not sel.any():
        return float("nan")
    return float(np.median(np.asarray(v, float)[sel]) * scale)


def report_dump(tag, hand, alpha, seed, st, meta, out):
    out.append("## %s  (hand=%s α=%g seed=%s) — rounds=%d · 유효 상태 %d/%d · %s"
               % (tag, hand, alpha if alpha is not None else float(meta["alpha"]), seed,
                  meta["rounds"], meta["n_states"], meta["rounds"] * meta["n_main"], meta["cfg"]))
    out.append("- G(초기 파지 성공률)=%.3f · 원 DemoGrasp SR(main, %d rounds)=%.3f · 창미완료(g1·done0)=%.3f · 미시작(g0=0)=%.3f"
               % (meta["G_rate"], meta["sr_rounds"], meta["sr_main"], meta["g1_incomplete"], meta["g0_zero"]))
    cf = st["ctrl_fin"]
    gate_hold = float(st["ctrl_y_hold"][cf].mean()) if cf.any() else float("nan")
    gate_pose = float(st["ctrl_y_pose"][cf].mean()) if cf.any() else float("nan")
    gate_ep = med_wfinite(st["ctrl_ep_end"], cf, 1000.0)
    ok = (gate_hold >= GATE_HOLD) and (gate_ep <= GATE_EP_MM)
    out.append("- 게이트: control Y_hold=%.3f(≥%.2f) · ep_end med=%.1fmm(≤%.0fmm) · Y_pose=%.3f → %s"
               % (gate_hold, GATE_HOLD, gate_ep, GATE_EP_MM, gate_pose, "PASS" if ok else "FAIL"))
    rd = [macro(st["p_y_hold"][dd][st["p_fin"][dd]], st["objs"][st["p_fin"][dd]]) for dd in range(6)]
    rdp = [macro(st["p_y_pose"][dd][st["p_fin"][dd]], st["objs"][st["p_fin"][dd]]) for dd in range(6)]
    out.append("- R_d(hold): " + " ".join("%s %.3f" % (DIRN[dd], rd[dd]) for dd in range(6))
               + "  |  R_mean=%.3f" % float(np.mean(rd)))
    out.append("- R_d(pose): " + " ".join("%s %.3f" % (DIRN[dd], rdp[dd]) for dd in range(6))
               + "  |  R_mean=%.3f" % float(np.mean(rdp)))
    ap_h, nh0 = r_all(st, "p_y_hold")
    ap_p, _ = r_all(st, "p_y_pose")
    lo, hi = boot_ci(ap_h, st["objs"], np.random.default_rng(BOOTSEED))
    out.append("- R_all(hold)=%.3f [95%%CI %.3f, %.3f] · R_all(pose)=%.3f · 제외(done probe 0개)=%d"
               % (macro(ap_h, st["objs"]), lo, hi, macro(ap_p, st["objs"]), nh0))
    out.append("- 방향별 ep_max med(mm): " + " ".join(
        "%s %.1f" % (DIRN[dd], med_wfinite(st["p_ep_max"][dd], st["p_fin"][dd], 1000.0)) for dd in range(6)))
    out.append("- 방향별 eR_max med(deg): " + " ".join(
        "%s %.1f" % (DIRN[dd], med_wfinite(st["p_eR_max"][dd], st["p_fin"][dd], RAD2DEG)) for dd in range(6)))
    out.append("")


def pool(states):
    """여러 dump 의 상태 배열을 이어붙인다."""
    return dict(
        objs=np.concatenate([s["objs"] for s in states]),
        ctrl_fin=np.concatenate([s["ctrl_fin"] for s in states]),
        ctrl_y_hold=np.concatenate([s["ctrl_y_hold"] for s in states]),
        ctrl_y_pose=np.concatenate([s["ctrl_y_pose"] for s in states]),
        ctrl_ep_end=np.concatenate([s["ctrl_ep_end"] for s in states]),
        **{k: np.concatenate([s[k] for s in states])
           for k in ("p_fin", "p_y_hold", "p_y_pose", "p_ep_max", "p_eR_max")})


def main():
    ap = argparse.ArgumentParser(description="posthold 분기 결과 분석 (EXP_POSTLIFT §1)")
    ap.add_argument("paths", nargs="+", help="runs/posthold/<tag>/branch.npz (glob 가능)")
    ap.add_argument("--out", default=None, help="markdown 을 파일로도 저장")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files += sorted(glob.glob(p))
    if not files:
        raise SystemExit("[36] 입력 npz 없음: %s" % args.paths)

    out = ["# posthold 결과 (판정규칙 = EXP_POSTLIFT §1·§4, 사전등록)", ""]
    groups = {}
    for path in files:
        tag, hand, alpha, seed = parse_tag(path)
        st, meta = load_states(path)
        if hand is None:
            out.append("(경고) tag '%s' 가 <hand>_a<alpha>_s<seed> 규약과 안 맞는다 — 요약 표에서 제외" % tag)
        report_dump(tag, hand, alpha, seed, st, meta, out)
        if hand is not None and st["objs"].size:
            g = groups.setdefault((hand, alpha), {"st": [], "gw": []})
            g["st"].append(st)
            g["gw"].append((meta["G_rate"], meta["rounds"] * meta["n_main"]))

    out.append("# 요약 (hand×alpha, seed 통합, 물체 macro 평균)")
    out.append("")
    head = ["hand", "α", "G"] + DIRN + ["R_mean", "R_all[CI]", "R_all_pose", "gate(hold/ep_mm/pose)"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for (hand, alpha), g in sorted(groups.items()):
        st = pool(g["st"])
        rd = [macro(st["p_y_hold"][dd][st["p_fin"][dd]], st["objs"][st["p_fin"][dd]]) for dd in range(6)]
        ap_h, _ = r_all(st, "p_y_hold")
        ap_p, _ = r_all(st, "p_y_pose")
        lo, hi = boot_ci(ap_h, st["objs"], np.random.default_rng(BOOTSEED))
        cf = st["ctrl_fin"]
        gate = "%.2f/%s/%.2f" % (
            st["ctrl_y_hold"][cf].mean() if cf.any() else float("nan"),
            ("%.1f" % med_wfinite(st["ctrl_ep_end"], cf, 1000.0)) if cf.any() else "NA",
            st["ctrl_y_pose"][cf].mean() if cf.any() else float("nan"))
        wtot = sum(w for _, w in g["gw"])
        grate = sum(r * w for r, w in g["gw"]) / wtot if wtot else float("nan")
        row = [hand, "%g" % alpha, "%.3f" % grate] + ["%.3f" % x for x in rd] + [
            "%.3f" % float(np.mean(rd)),
            "%.3f [%.3f,%.3f]" % (macro(ap_h, st["objs"]), lo, hi),
            "%.3f" % macro(ap_p, st["objs"]), gate]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("게이트 기준: control Y_hold ≥ %.2f · ep_end med ≤ %.0f mm (EXP_POSTLIFT §2). CI=물체 cluster bootstrap %d회."
               % (GATE_HOLD, GATE_EP_MM, NBOOT))

    text = "\n".join(out) + "\n"
    print(text, end="")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print("[36] wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
