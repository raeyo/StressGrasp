#!/usr/bin/env python3
"""graspstress 집계 — runs/{d1,stress}/*/eval.log → results/stage0_summary.{json,md}

셀 규약:  d1     : <ckpt>__<objset>__tau<N>__seed<S>
          stress : <ckpt>__<objset>__tau<N>__m<M>__g<G>__seed<S>
SR = 10 round 평균. 기준선(tau0=50, m=1, g=0) 대비 낙폭 ΔSR 을 낸다 (PROTOCOL §5).
"""
import json, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEP_S = 0.3333  # dt 0.01667 x decimation 20

def rounds(log):
    return [float(x) for x in re.findall(r"success rate: ([0-9.]+)", log.read_text(errors="ignore"))]

cells = []
for log in sorted(ROOT.glob("runs/*/*/eval.log")):
    name, axis = log.parent.name, log.parent.parent.name
    if axis.startswith("_"):      # runs/_invalid_* = 무효 격리 (인용 금지)
        continue
    srs = rounds(log)
    if len(srs) < 10:
        continue
    m = re.match(r"(.+?)__(.+?)__tau(\d+)__m([0-9.]+)__g([0-9.]+)(?:__(pre|grasp|post|all))?__seed(\d+)$", name)
    if m:
        ck, obj, tau, mass, extg = m.group(1), m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5))
        phase, seed = (m.group(6) or "post"), int(m.group(7))
    else:
        m = re.match(r"(.+?)__(.+?)__tau(\d+)__seed(\d+)$", name)
        if not m:
            continue
        ck, obj, tau, mass, extg = m.group(1), m.group(2), int(m.group(3)), 1.0, 0.0
        phase, seed = "post", int(m.group(4))
    cells.append(dict(axis=axis, cell=name, ckpt=ck, objset=obj, tau=tau, mass=mass,
                      ext_g=extg, phase=phase, seed=seed, n=len(srs), sr=sum(srs)/len(srs),
                      round_std=statistics.pstdev(srs)))

if not cells:
    sys.exit("no completed cells")

def base_of(c):
    for b in cells:
        if (b["ckpt"], b["objset"], b["seed"]) == (c["ckpt"], c["objset"], c["seed"]) \
           and b["tau"] == 50 and b["mass"] == 1.0 and b["ext_g"] == 0.0:
            return b
    return None

for c in cells:
    b = base_of(c)
    c["sr_base"] = b["sr"] if b else None
    c["delta"] = (b["sr"] - c["sr"]) if b else None

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "stage0_summary.json").write_text(
    json.dumps(sorted(cells, key=lambda c: (c["objset"], c["tau"], c["mass"], c["ext_g"])),
               indent=2, ensure_ascii=False))

def tbl(sel, title, keycol, keyfn):
    rows = sorted([c for c in cells if sel(c)], key=lambda c: (c["objset"], keyfn(c)))
    if not rows:
        return []
    out = ["", f"### {title}", "",
           f"| objset | {keycol} | SR | ΔSR vs 기준 | round-std |", "|---|---|---|---|---|"]
    for c in rows:
        d = f"**{c['delta']:+.4f}**" if c["delta"] is not None else "—"
        out.append(f"| {c['objset']} | {keyfn(c)} | {c['sr']:.4f} | {d} | {c['round_std']:.4f} |")
    return out

md = ["# Stage 0 — 낙폭 측정 요약", "",
      f"셀 {len(cells)}개 · 1 step = {STEP_S}s · 기준선 = tau50 / mass×1 / ext_g 0", ""]
md += tbl(lambda c: c["mass"] == 1 and c["ext_g"] == 0, "D1 유지 시간", "tau (step / s)",
          lambda c: f"{c['tau']} / {c['tau']*STEP_S:.1f}s")
md += tbl(lambda c: c["mass"] != 1, "D2 하중 (질량 배율)", "mass x", lambda c: c["mass"])
PH_KO = {"pre": "전", "grasp": "중", "post": "후"}
for ph in ("pre", "grasp", "post"):
    md += tbl(lambda c, ph=ph: c["ext_g"] != 0 and c["phase"] == ph,
              "D3 외력 — 파지 " + PH_KO[ph], "ext_g (무게 배수)", lambda c: c["ext_g"])
md += ["", "## 판정 (PROTOCOL §5)", ""]
ds = [c["delta"] for c in cells if c["delta"] is not None]
mx = max(ds) if ds else 0.0
# ★ 사전 등록(PROTOCOL §5)은 임계값만 고정했고 **물리적으로 방어 가능한 강도 범위는 고정하지
#   않았다.** 이건 프로토콜의 공백이다. 임계값을 사후에 바꾸지 않는다 — 대신 사실을 분리해 적는다.
sub = [c["delta"] for c in cells
       if c["delta"] is not None and c["ext_g"] <= 10 and c["mass"] <= 16]
mxs = max(sub) if sub else 0.0
rule = lambda v: ("F1 — 낙폭 없음" if v < 0.05 else
                  "판정 유보" if v < 0.10 else "문제 실재")
md += [f"- **전 구간** 최대 낙폭 ΔSR = **{mx:+.4f}** → 규칙상 **{rule(mx)}**",
       f"- **조작 현실 범위**(ext_g ≤ 10 = 10×중력 측방가속, mass ≤ ×16) 최대 낙폭 "
       f"ΔSR = **{mxs:+.4f}** → 규칙상 **{rule(mxs)}**",
       "",
       "⚠️ **프로토콜 공백**: §5 는 임계값만 사전 고정했고 **강도 범위를 고정하지 않았다.**",
       "위 두 줄은 같은 데이터의 두 읽기이며, 어느 범위를 유효한 조작 조건으로 볼지는",
       "**측정이 아니라 범위 정의의 문제**다. 임계값을 사후에 수정하지 않았다.", ""]
(ROOT / "results" / "stage0_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
