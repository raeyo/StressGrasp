#!/usr/bin/env python3
"""실패 조건 지도 — runs/grid/ · runs/student/ · runs/stress/ 를 한 표로.

판정은 하지 않는다. **조건별 SR 과 clean 대비 낙폭만** 낸다 (사용자 지시 2026-09-22).
태그 규약:
  grid    : <ckpt>__<hand>__<objset>__m<M>__g<G>__<phase>d<D>__s<S>__seed<N>
  student : <run>__<hand>__<objset>__H<list>__m<M>__g<G>__<phase>d<D>__s<S>__seed<N>
"""
import json, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID = re.compile(r"^(?P<pol>.+?)__(?P<hand>[^_]+(?:_[^_]+)*?)__(?P<obj>ours_[SML]|[^_]+)__"
                  r"m(?P<m>[0-9.]+)__g(?P<g>[0-9.]+)__(?P<ph>pre|grasp|post|all)d(?P<d>\d+)__"
                  r"s(?P<s>[0-9.]+)__seed(?P<seed>\d+)$")

def rounds(p):
    t = p.read_text(errors="ignore")
    return [float(x) for x in re.findall(r"success rate: ([0-9.]+)", t)]

cells = []
for log in sorted(ROOT.glob("runs/grid/*/eval.log")):
    m = GRID.match(log.parent.name)
    if not m:
        continue
    srs = rounds(log)
    if len(srs) < 10:
        continue
    d = m.groupdict()
    cells.append(dict(policy=d["pol"], hand=d["hand"], objset=d["obj"], mass=float(d["m"]),
                      ext_g=float(d["g"]), phase=d["ph"], dose=int(d["d"]), shift_cm=float(d["s"]),
                      seed=int(d["seed"]), sr=sum(srs) / len(srs),
                      std=statistics.pstdev(srs), mode="teacher"))

# RobustDexGrasp: "==== RobustDexGrasp in DemoGrasp env: mean SR = x +/- y"
RDX = re.compile(r"^rdx__(?P<hand>[^_]+(?:_[^_]+)*?)__(?P<obj>.+?)__m(?P<m>[0-9.]+)__g(?P<g>[0-9.]+)__"
                 r"(?P<ph>pre|grasp|post|all)d(?P<d>\d+)__s(?P<s>[0-9.]+)__seed(?P<seed>\d+)$")
for log in sorted(ROOT.glob("runs/rdx/*/eval.log")):
    m = RDX.match(log.parent.name)
    if not m:
        continue
    hit = re.findall(r"mean SR = ([0-9.]+) \+/- ([0-9.]+)", log.read_text(errors="ignore"))
    if not hit:
        continue
    d = m.groupdict()
    cells.append(dict(policy="RobustDexGrasp", hand=d["hand"], objset=d["obj"], mass=float(d["m"]),
                      ext_g=float(d["g"]), phase=d["ph"], dose=int(d["d"]), shift_cm=float(d["s"]),
                      seed=int(d["seed"]), sr=float(hit[-1][0]), std=float(hit[-1][1]),
                      mode="closed-loop 5Hz"))

# student: "=== OPEN-LOOP  SR = x +/- y" / "=== CLOSED-LOOP H=8 SR = ..."
for log in sorted(ROOT.glob("runs/student/*/eval.log")):
    name = log.parent.name
    txt = log.read_text(errors="ignore")
    base = re.sub(r"__H[0-9\-]+", "", name)
    m = GRID.match(base)
    for tag, sr in re.findall(r"=== (.+?)\s+SR = ([0-9.]+)", txt):
        d = m.groupdict() if m else {}
        cells.append(dict(policy=d.get("pol", name), hand=d.get("hand", "?"),
                          objset=d.get("obj", "?"), mass=float(d.get("m", 1)),
                          ext_g=float(d.get("g", 0)), phase=d.get("ph", "post"),
                          dose=int(d.get("d", 0)), shift_cm=float(d.get("s", 0)),
                          seed=int(d.get("seed", 42)), sr=float(sr), std=0.0,
                          mode=tag.strip()))

def is_clean(c):
    return c["mass"] == 1 and c["ext_g"] == 0 and c["shift_cm"] == 0

for c in cells:
    base = next((b for b in cells
                 if is_clean(b) and (b["policy"], b["hand"], b["objset"], b["seed"], b["mode"])
                 == (c["policy"], c["hand"], c["objset"], c["seed"], c["mode"])), None)
    c["sr_clean"] = base["sr"] if base else None
    c["delta"] = (c["sr"] - base["sr"]) if base else None   # 낙폭 = 음수

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "failmap.json").write_text(json.dumps(cells, indent=2, ensure_ascii=False))

def cond(c):
    if is_clean(c):
        return "clean"
    if c["shift_cm"]:
        return f"물체변위 {c['shift_cm']:.0f}cm (파지 전)"
    if c["mass"] != 1:
        return f"질량 x{c['mass']:.0f} ({c['mass']*0.0309:.2f}kg)"
    ko = {"pre": "파지 전", "grasp": "파지 중", "post": "파지 후"}[c["phase"]]
    return f"외력 {c['ext_g']:.0f}g @{ko}" + (f" /{c['dose']}step" if c["dose"] else "")

md = ["# 실패 조건 지도", "", "> 판정 없음. 조건별 SR 과 clean 대비 낙폭만. clean 은 각 (정책,손,물체셋) 자기 자신 기준.",
      "", "| 정책 | 손 | 물체셋 | 모드 | 조건 | SR | ΔSR |", "|---|---|---|---|---|---|---|"]
for c in sorted(cells, key=lambda c: (c["policy"], c["hand"], c["objset"], c["mode"],
                                      c["shift_cm"], c["mass"], c["ext_g"])):
    dd = f"**{c['delta']:+.4f}**" if c["delta"] is not None else "—"
    md.append(f"| {c['policy']} | {c['hand']} | {c['objset']} | {c['mode']} | {cond(c)} "
              f"| {c['sr']:.4f} | {dd} |")
(ROOT / "results" / "failmap.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
