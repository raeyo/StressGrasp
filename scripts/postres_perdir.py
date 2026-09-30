#!/usr/bin/env python3
"""★ 사후·탐색적 — POSTRES/WORLDLOAD 데이터를 **방향 특정** 지표로 다시 읽는다.
동기: R_all_pose(6방향 전부)는 회전에 대해 원리적으로 무력하다 — 약축을 다른 방향으로 옮길 뿐
6개를 다 피할 수는 없다. 손목 방향이 축이 되려면 목적함수가 **방향 특정**이어야 한다 (운반 = I-26).
지표 = g0 ∧ done[d] ∧ ypose[d] (방향 d 하나). 세계 고정 실행에서는 d 가 세계축이다 (−z = 중력 정렬).
판정 아님 — 사전등록 없음. 다음 실험의 가설로만 쓴다.
"""
import sys, numpy as np
DIRS = ["+x", "-x", "+y", "-y", "+z", "-z"]
path, hand = sys.argv[1], sys.argv[2]
d = np.load(path)
scene, obj, slot, arm, lev = (d[k].astype(int) for k in ("scene", "obj", "slot", "arm", "lev"))
names = ["ZERO"] + [str(x) for x in d["meta_armnames"][1:]]
key = scene * 10000 + obj
idx = {}
for i in range(len(key)): idx.setdefault((key[i], slot[i]), i)
units = np.unique(key)
rng = np.random.RandomState(0)
print("== %s · 방향 특정 지표 (사후·탐색적) ==" % hand)
for dd in range(6):
    def y(r): return ((d["g0_"+r] > 0) & (d["done_"+r][:, dd] > 0) & (d["ypose_"+r][:, dd] > 0.5)).astype(float)
    A, B = y("A"), y("B")
    line = []
    for ai in range(1, 4):
        g = []
        for li in range(3):
            ss = [int(s) for s in np.unique(slot) if s > 0 and arm[slot == s][0] == ai and lev[slot == s][0] == li]
            te, bu = [], []
            for u in units:
                i0 = idx.get((u, 0))
                if i0 is None: continue
                ii = np.array([idx[(u, s)] for s in ss if (u, s) in idx])
                if not len(ii): continue
                te.append((A[i0]+B[i0])/2); bu.append(B[ii[int(np.argmax(A[ii]))]])
            g.append(float(np.mean(bu) - np.mean(te)))
        line.append("%s %s" % (names[ai], "/".join("%+.3f" % x for x in g)))
    tea = float(np.mean([(A[idx[(u,0)]]+B[idx[(u,0)]])/2 for u in units if (u,0) in idx]))
    # ROT 레벨 간 범위 = 대칭이 깨졌는지의 지표
    gr = [float(x) for x in line[1].split()[1].split("/")]
    print("   %-3s teacher %.3f | %s | ★ROT 범위 %.3f" % (DIRS[dd], tea, " · ".join(line), max(gr)-min(gr)))
