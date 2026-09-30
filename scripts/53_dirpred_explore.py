#!/usr/bin/env python3
"""DIRPRED 탐색적 분석 — ★ 게이트 판정이 아니다 (사전등록 외, RESULTS 에 '탐색적' 으로만 병기).
앙상블 std 대신 예측 엔트로피(p(1-p)) 를 불확실성으로 썼을 때 A2/A3 형태의 비율이 어떻게 되는가.
usage: python3 scripts/53_dirpred_explore.py runs/critic/_dp/pred_<tag>_<feat>.npz [P|PT|...]
"""
import sys, numpy as np
z = np.load(sys.argv[1]); nm = sys.argv[2] if len(sys.argv) > 2 else "P"
Y, nd = z["Y"], z["nondeg"]
pm, ps = z[nm + "_pm"], z[nm + "_ps"]
err = np.abs(pm - Y)
def quart_ratio(u, e):
    q1, q3 = np.percentile(u, [25, 75]); return e[u >= q3].mean() / max(e[u <= q1].mean(), 1e-9)
ent = (pm * (1 - pm))
print("== %s  %s  (홀드아웃 n=%d, 비퇴화 축 %s)" % (sys.argv[1], nm, len(Y), np.where(nd)[0].tolist()))
print("  [A2형] 표본 평균:   앙상블 std 비 %.2f | 엔트로피 비 %.2f" % (quart_ratio(ps.mean(1), err.mean(1)), quart_ratio(ent.mean(1), err.mean(1))))
print("  [A2형] 비퇴화 축별: " + " ".join("std %.2f/ent %.2f" % (quart_ratio(ps[:, i], err[:, i]), quart_ratio(ent[:, i], err[:, i])) for i in np.where(nd)[0]))
if nm + "_pso" in z:
    pso, pmo = z[nm + "_pso"], z[nm + "_pmo"]
    ento = pmo * (1 - pmo)
    print("  [A3형] 가림/비가림: 앙상블 std 비 %.2f | 엔트로피 비 %.2f | 가림 오차/비가림 오차 %.2f"
          % (pso.mean() / ps.mean(), ento.mean() / ent.mean(), np.abs(pmo - Y).mean() / err.mean()))
    # 가림에 의한 예측 변화량 자체가 '관측 불충분' 신호인가: |pmo - pm| 상위 25% 의 오차
    dif = np.abs(pmo - pm).mean(1)
    print("  [참고] 가림 전후 예측 변화량 상위25%% 표본의 가림 오차 / 하위25%% = %.2f" % quart_ratio(dif, np.abs(pmo - Y).mean(1)))
