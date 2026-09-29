"""graspstress residual RL — DemoGrasp 레퍼런스 행동 위에 Δ 를 얹어 grip 구간 외력 마진을 올린다.

설계 = docs/EXP_RESIDUAL.md. 우산 README §1: 남의 코드(DemoGrasp)는 고치지 않는다.
`algo.ppo_onestep.ppo.PPO.run` 을 런타임에 가로채 우리 학습 루프를 돌린다 (env 생성·설정은 원본 그대로).

■ 행동
  a_t = compute_reference_actions()  +  Δ(o_t)          ← tasks/grasp.py:1313 이 주입점
  Δ = [EE 위치 3, 손 관절 6].  ±1 cm / ±0.15 (정규화 단위) 로 클램프.
  ★ grip phase 에서만 적용한다. 접근·들어올림은 원 레퍼런스 그대로 둔다.
  ★ Δ 는 main env 의 관측으로 한 번 계산해 **그룹 전체에 같은 값**을 쓴다
    (그림자는 같은 컨트롤러로 굴러야 마진 측정의 의미가 유지된다).

■ 보상 (두 변형 = 이 실험의 개입 축)
  분기 지평 H 스텝마다 그림자 env 의 물체 변위 d_i (i = ±xyz 6방향) 를 읽어
    r_dir_i = exp(-d_i / tau)
    MIN : r = min_i r_dir_i      (= 최약축을 올린다. ε 에 해당)
    SUM : r = mean_i r_dir_i     (= 평균 저항을 올린다)
  + 종단에 원 성공판정 보너스 W_SUCC * success  (lift 를 잃지 않게 한다)

환경변수: GS_RESIDUAL=min|sum  ·  GS_RES_ITERS  ·  GS_RES_TAG  ·  GS_RES_TAU(mm)
          GS_RES_WSUCC  ·  GS_RES_LR  ·  GS_RES_EVAL=1 (학습 없이 ckpt 평가)  ·  GS_RES_CKPT
"""
import os

_MODE = os.environ.get("GS_RESIDUAL", "")           # "" = 무동작
_ON = _MODE in ("min", "sum", "zero", "plan_min", "plan_sum",
                "bmin", "bsum", "plan_bmin", "plan_bsum", "sweep", "oracle", "replay")
_SWEEP = os.environ.get("GS_RES_SWEEP", "hand")     # hand | posz | posx — 무엇을 쓸어볼지
_ALLPH = os.environ.get("GS_RES_ALLPHASE", "") not in ("", "0")   # grip 게이팅 해제(접근부터 적용)
_BIN = (_MODE.split("_")[-1].startswith("b")
        or os.environ.get("GS_RES_BIN", "") not in ("", "0"))   # 이진 보상 변형
_THETA = float(os.environ.get("GS_RES_THETA", "20") or 20) / 1000.0   # 이진 임계 (mm -> m)
_PLAN = _MODE.startswith("plan")          # t=0 계획(12-D edit)에 residual 을 얹는 변형
_AGG = "min" if _MODE.endswith("min") else "mean"
_SHAPE = os.environ.get("GS_RES_SHAPE", "1" if _MODE.startswith("plan") else "0") not in ("", "0")
_DPLAN = float(os.environ.get("GS_RES_DPLAN", "0.2") or 0.2)   # 계획 residual 한계 ([-1,1] 공간)
_ITERS = int(os.environ.get("GS_RES_ITERS", "150") or 150)
_TAG = os.environ.get("GS_RES_TAG", "res")
_TAU = float(os.environ.get("GS_RES_TAU", "5") or 5) / 1000.0     # mm -> m
_WSUCC = float(os.environ.get("GS_RES_WSUCC", "1.0") or 1.0)
_LR = float(os.environ.get("GS_RES_LR", "3e-4") or 3e-4)
_OUT = os.environ.get("GS_RES_OUT", "")
_EVAL = os.environ.get("GS_RES_EVAL", "") not in ("", "0")
_CKPT = os.environ.get("GS_RES_CKPT", "")
_DPOS = float(os.environ.get("GS_RES_DPOS", "0.01") or 0.01)      # EE 위치 residual 한계 (m)
_DHAND = float(os.environ.get("GS_RES_DHAND", "0.15") or 0.15)    # 손 관절 residual 한계 (정규화)


def _reward(dd, torch):
    """dd = [6, n] 변위(m) -> [n] 보상.
    ★ 이진(원안): 움직였으면 0, 안 움직였으면 1.  기댓값이 곧 '버틸 확률' 이라
      policy gradient 가 그것을 직접 추정한다. 연속형 τ/(τ+d) 는 최약축이 항상 파국이라
      r≈상수(기울기 0)가 되어 학습이 서지 않았다 (RESULTS_RESIDUAL §2)."""
    if _BIN:
        per = (dd < _THETA).float()
    else:
        per = _TAU / (_TAU + dd)
    r = per.min(0).values if _AGG == "min" else per.mean(0)
    return r, per


def _build():
    import numpy as np
    import torch
    import torch.nn as nn
    from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul

    # ---------------------------------------------------------------- 관측
    def priv_obs(env):
        """특권 관측 — 마진과 물리적으로 관계있는 것만 모은 압축 벡터 (손바닥 좌표계)."""
        n = env._br_n_main
        pq = env.palm_rot[:n]
        pp = env.palm_pos[:n]
        inv = quat_conjugate(pq)
        F = env.fingertip_pos.shape[1]

        def to_palm(v):                       # [n, k, 3] -> 손바닥 좌표
            k = v.shape[1]
            q = inv.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
            return quat_apply(q, v.reshape(-1, 3)).reshape(n, k * 3)

        obj_rel = to_palm((env.object_pos[:n] - pp).unsqueeze(1))
        obj_q = quat_mul(inv, env.object_rot[:n])
        ft_rel = to_palm(env.fingertip_pos[:n] - pp.unsqueeze(1))
        cf = to_palm(env.contact_force_sensors[:n]) * 0.1
        lv = to_palm(env.object_linvel[:n].unsqueeze(1))
        av = to_palm(env.object_angvel[:n].unsqueeze(1))
        lo = env.robot_dof_lower_limits[env.active_hand_dof_indices]
        hi = env.robot_dof_upper_limits[env.active_hand_dof_indices]
        hq = (2.0 * (env.robot_dof_pos[:n][:, env.active_hand_dof_indices] - lo) / (hi - lo) - 1.0)
        t = env.progress_buf[:n].float().unsqueeze(-1) / float(env.max_episode_length)
        age = (env.progress_buf[:n].long() - env._br_t_grip).clamp(min=0).float().unsqueeze(-1) / 20.0
        grip = env._br_grip_mask()[:n].float().unsqueeze(-1)
        feats = [hq, obj_rel, obj_q, ft_rel, cf, lv, av, t, age, grip]
        if _SHAPE and hasattr(env, "transformed_pcl"):
            # 물체 형상/크기 — 계획을 고치려면 자세만으론 부족하다 (손바닥 좌표계 PCL 요약)
            pc = env.transformed_pcl[:n]                       # [n, P, 3]
            pcp = to_palm(pc - pp.unsqueeze(1)).reshape(n, -1, 3)
            feats += [pcp.mean(1), pcp.std(1), pcp.min(1).values, pcp.max(1).values]
        return torch.cat(feats, dim=-1)

    class Net(nn.Module):
        def __init__(self, din, dact):
            super().__init__()
            def mlp(o):
                return nn.Sequential(nn.Linear(din, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                                     nn.Linear(128, o))
            self.pi, self.vf = mlp(dact), mlp(1)
            self.log_std = nn.Parameter(torch.full((dact,), -1.0))
            nn.init.zeros_(self.pi[-1].weight); nn.init.zeros_(self.pi[-1].bias)   # ★ Δ=0 에서 출발

        def dist(self, o):
            return torch.distributions.Normal(self.pi(o), self.log_std.exp())

    class RunNorm:
        def __init__(self, d, dev):
            self.m = torch.zeros(d, device=dev); self.v = torch.ones(d, device=dev); self.n = 1e-4

        def __call__(self, x, update=True):
            if update:
                bn = x.shape[0]; bm = x.mean(0); bv = x.var(0, unbiased=False)
                d = bm - self.m; tot = self.n + bn
                self.m = self.m + d * bn / tot
                self.v = (self.v * self.n + bv * bn + d.pow(2) * self.n * bn / tot) / tot
                self.n = tot
            return ((x - self.m) / (self.v.sqrt() + 1e-5)).clamp(-8, 8)

    return priv_obs, Net, RunNorm, np, torch, nn


def _run(ppo_self):
    priv_obs, Net, RunNorm, np, torch, nn = _build()
    env = ppo_self.vec_env
    dev = env.device
    n = env._br_n_main
    T = int(env.max_episode_length)
    out = _OUT or os.path.join(os.environ.get("GS_ROOT", "."), "runs", "residual", _TAG)
    os.makedirs(out, exist_ok=True)

    env.reset_idx(torch.arange(env.num_envs, device=dev))
    din = priv_obs(env).shape[-1]
    dact = 3 + env.num_active_hand_dofs
    if _CKPT and os.path.exists(_CKPT):
        _ck0 = torch.load(_CKPT, map_location="cpu")
        dact = int(_ck0.get("dact", dact))
    net = Net(din, dact).to(dev)
    norm = RunNorm(din, dev)
    if _CKPT:
        ck = torch.load(_CKPT, map_location=dev)
        net.load_state_dict(ck["net"])
        norm.m, norm.v, norm.n = ck["norm_m"].to(dev), ck["norm_v"].to(dev), float(ck["norm_n"])
        print("[res] loaded", _CKPT, flush=True)

    def _save_ck():
        torch.save({"net": net.state_dict(), "norm_m": norm.m.cpu(), "norm_v": norm.v.cpu(),
                    "norm_n": norm.n, "mode": _MODE, "din": din, "dact": dact},
                   os.path.join(out, "policy.pt"))
    opt = torch.optim.Adam(net.parameters(), lr=_LR)
    print("[res] mode=%s%s obs=%d act=%d n_main=%d T=%d iters=%d %s wsucc=%.2f out=%s"
          % (_MODE, " [BINARY]" if _BIN else " [cont]", din, dact, n, T, _ITERS,
             ("theta=%.0fmm" % (_THETA * 1000)) if _BIN else ("tau=%.1fmm" % (_TAU * 1000)),
             _WSUCC, out), flush=True)

    src = env._br_src
    G = env._br_G
    log = open(os.path.join(out, "train.csv"), "a")
    log.write("iter,ret,margin_min,margin_mean,succ,nwin,dmax_mm,dmean_mm,dstd\n"); log.flush()

    if _MODE == "replay":
        # ★ 오라클 해가 **다른 초기자세로 전이되는가**. 전이되면 "물체 수준에서 학습 가능",
        #   안 되면 "장면마다 다시 풀어야 하는 문제" 다.
        n_obj = env._br_n_obj
        z = np.load(os.environ["GS_RES_ORACLE"])
        ba = torch.tensor(z["best_action"], dtype=torch.float, device=dev)      # [n_obj, 12]
        te = torch.tensor(z["teacher"], dtype=torch.float, device=dev)
        pose0 = torch.tensor(z["pose0"], dtype=torch.float, device=dev)
        fix = os.environ.get("GS_RES_FIXPOSE", "") not in ("", "0")
        use_teacher = os.environ.get("GS_RES_USETEACHER", "") not in ("", "0")
        A = te if use_teacher else ba
        allenv = torch.arange(env.num_envs, device=dev)
        obj = env._br_obj[:n]
        HO = []; SC = []
        print("[res] REPLAY  %s  자세=%s  n_obj=%d n_main=%d"
              % ("teacher edit" if use_teacher else "오라클 해", "고정" if fix else "랜덤", n_obj, n), flush=True)
        for rd in range(10):
            if fix:
                pp = pose0.repeat(int(np.ceil(env.num_envs / n_obj)), 1)[:env.num_envs]
                env.reset_idx(allenv, object_init_pose=pp.cpu().numpy())
            else:
                env.reset_idx(allenv)
            plan = torch.zeros(env.num_envs, A.shape[-1], device=dev)
            plan[:n] = A[obj]
            env.generate_reaching_plan_idx(allenv, actions=plan[env._br_src])
            acc = torch.zeros(n, device=dev); nw = torch.zeros(n, device=dev)
            for t in range(T):
                env.step(env.compute_reference_actions())
                if getattr(env, "_br_active_win", False) and env._br_win == 0:
                    p_ = env.root_state_tensor[env.object_indices, 0:3]
                    dd = torch.stack([torch.norm(p_[g2 * n:(g2 + 1) * n] - p_[:n], dim=-1)
                                      for g2 in range(2, G)], 0)
                    _, per = _reward(dd, torch)
                    gm = env._br_grip_mask()[:n].float()
                    acc += per.mean(0) * gm; nw += gm
                    HO.append(per[:, gm > 0].mean(1).detach().cpu().numpy() if bool((gm > 0).any()) else np.zeros(6))
            sc = (env.successes[:n] > 0.5).float()
            SC.append(float(sc.mean()))
            print("[res]  round %d  liftSR=%.4f  hold평균=%.4f" % (rd, float(sc.mean()),
                  float((acc / nw.clamp(min=1))[nw > 0].mean())), flush=True)
        h = np.mean(np.stack(HO), 0)
        print("[res] === REPLAY 결과: liftSR=%.4f  hold/dir %s  평균 %.4f ==="
              % (float(np.mean(SC)), " ".join("%s=%.3f" % (d, v) for d, v in
                 zip(["+x", "-x", "+y", "-y", "+z", "-z"], h)), float(h.mean())), flush=True)
        raise SystemExit(0)

    if _MODE == "oracle":
        # ★ 물체별 오라클 상한 — 학습 없이 CEM 으로 "이 장면에서 최선의 12-D edit" 을 찾는다.
        #   목적 = 개입 게이트를 **도달 가능한 범위** 위에서 정하기 위함 (RESULTS_RESIDUAL §9).
        #   물체 초기자세를 고정해 같은 장면에서 후보만 비교한다 (평가 잡음 제거).
        n_obj = env._br_n_obj
        R = n // n_obj                                   # 물체당 후보 수
        dp = 6 + env.num_active_hand_dofs
        allenv = torch.arange(env.num_envs, device=dev)
        gens = int(os.environ.get("GS_RES_GENS", "20") or 20)
        elite = max(2, R // 4)
        fixpose = os.environ.get("GS_RES_FIXPOSE", "1") not in ("", "0")
        cur = env.reset_idx(allenv)["obs"]
        pose0 = env.root_state_tensor[env.object_indices, 0:7].clone()   # 안착된 자세
        pose0 = pose0[:n_obj].repeat(int(np.ceil(env.num_envs / n_obj)), 1)[:env.num_envs]
        with torch.no_grad():
            teach = ppo_self.actor_critic(cur, env.get_state(), inference=True)[:, :dp].clone()
        mu = teach[:n_obj].clone()                       # 물체별 평균 = teacher 의 edit 에서 출발
        sd = torch.full_like(mu, float(os.environ.get("GS_RES_SIG", "0.35") or 0.35))
        best = torch.full((n_obj,), -1.0, device=dev); best_a = mu.clone()
        base = None
        print("[res] ORACLE  n_obj=%d R=%d gens=%d dim=%d sigma0=%.2f  (자세=%s, CEM)"
              % (n_obj, R, gens, dp, float(sd[0, 0]), "고정" if fixpose else "랜덤(전이가능 상한)"), flush=True)
        for g in range(gens):
            cand = (mu.repeat(R, 1) + sd.repeat(R, 1) * torch.randn(n, dp, device=dev)).clamp(-1, 1)
            if g == 0:
                cand[:n_obj] = teach[:n_obj]             # 후보 0 = teacher (기준선을 항상 포함)
            if fixpose:
                env.reset_idx(allenv, object_init_pose=pose0.cpu().numpy())
            else:
                env.reset_idx(allenv)      # ★ 자세를 매 세대 새로 뽑는다 -> 자세-강건한 해만 살아남는다
            plan = torch.zeros(env.num_envs, dp, device=dev)
            plan[:n] = cand
            env.generate_reaching_plan_idx(allenv, actions=plan[env._br_src])
            hold_acc = torch.zeros(n, device=dev); nw = torch.zeros(n, device=dev)
            for t in range(T):
                env.step(env.compute_reference_actions())
                if getattr(env, "_br_active_win", False) and env._br_win == 0:
                    p_ = env.root_state_tensor[env.object_indices, 0:3]
                    dd = torch.stack([torch.norm(p_[gg * n:(gg + 1) * n] - p_[:n], dim=-1)
                                      for gg in range(2, G)], 0)
                    _, per = _reward(dd, torch)
                    gm = env._br_grip_mask()[:n].float()
                    hold_acc += per.mean(0) * gm; nw += gm
            succ = (env.successes[:n] > 0.5).float()
            score = torch.where(nw > 0, hold_acc / nw.clamp(min=1), torch.zeros_like(hold_acc))
            score = torch.where(succ > 0.5, score, torch.full_like(score, -1.0))   # lift 실패 = 탈락
            S = score.view(R, n_obj); C = cand.view(R, n_obj, dp)
            if g == 0:
                base = S[0].clone()
            top = S.topk(min(elite, R), dim=0).indices                 # [elite, n_obj]
            sel = torch.gather(C, 0, top.unsqueeze(-1).expand(-1, -1, dp))
            mu = sel.mean(0); sd = sel.std(0).clamp(min=0.05)
            b, bi = S.max(0)
            upd = b > best
            best = torch.where(upd, b, best)
            best_a[upd] = torch.gather(C, 0, bi.view(1, n_obj, 1).expand(1, -1, dp))[0][upd]
            ok = best > -0.5
            print("[res] gen %2d  best 평균=%.4f (유효 %d/%d)  세대최고 평균=%.4f  σ 평균=%.3f"
                  % (g, float(best[ok].mean()) if ok.any() else -1, int(ok.sum()), n_obj,
                     float(b[b > -0.5].mean()) if (b > -0.5).any() else -1, float(sd.mean())), flush=True)
        okb = base > -0.5
        print("\n[res] === 오라클 상한 ===", flush=True)
        print("[res] teacher(기준선) hold 평균 = %.4f  (lift 성공 물체 %d/%d)"
              % (float(base[okb].mean()), int(okb.sum()), n_obj), flush=True)
        ok = best > -0.5
        print("[res] 오라클 최선   hold 평균 = %.4f  (유효 %d/%d)" % (float(best[ok].mean()), int(ok.sum()), n_obj), flush=True)
        both = okb & ok
        print("[res] 같은 물체 짝지어: teacher %.4f -> 오라클 %.4f   (Δ=%+.4f, %.2f배)"
              % (float(base[both].mean()), float(best[both].mean()),
                 float((best - base)[both].mean()), float(best[both].mean() / max(float(base[both].mean()), 1e-9))), flush=True)
        np.savez(os.path.join(out, "oracle.npz"), best=best.cpu().numpy(), base=base.cpu().numpy(),
                 best_action=best_a.cpu().numpy(), teacher=teach[:n_obj].cpu().numpy(),
                 pose0=pose0[:n_obj].cpu().numpy())
        print("[res] saved", os.path.join(out, "oracle.npz"), flush=True)
        raise SystemExit(0)

    if _MODE == "sweep":
        # ★ 감사: 고정된 큰 Δ 를 replica 별로 다르게 넣고 마진이 **움직이기는 하는가** 를 본다.
        #    안 움직이면 학습 문제가 아니라 하네스/설계 문제다.
        n_obj = env._br_n_obj
        R = n // n_obj
        lev = torch.linspace(-1.0, 1.0, R, device=dev)          # replica 별 Δ 수준
        rep = (torch.arange(n, device=dev) // n_obj)
        c = lev[rep]                                             # [n]
        allenv = torch.arange(env.num_envs, device=dev)
        acc = {r: {"hold": [], "succ": []} for r in range(R)}
        print("[res] SWEEP(%s) R=%d levels=%s  (Δpos=±%.3fm, Δhand=±%.3f)"
              % (_SWEEP, R, [round(float(v), 2) for v in lev], _DPOS, _DHAND), flush=True)
        for rd in range(5):
            cur = env.reset_idx(allenv)["obs"]
            with torch.no_grad():
                plan = ppo_self.actor_critic(cur, env.get_state(), inference=True)
            env.generate_reaching_plan_idx(allenv, actions=plan)
            for t in range(T):
                ref = env.compute_reference_actions()
                grip = env._br_grip_mask()[:n].float()
                dpos = torch.zeros(n, 3, device=dev)
                dhand = torch.zeros(n, env.num_active_hand_dofs, device=dev)
                if _SWEEP == "hand":
                    dhand = c.unsqueeze(-1).expand_as(dhand) * _DHAND
                elif _SWEEP == "posz":
                    dpos[:, 2] = c * _DPOS
                elif _SWEEP == "posx":
                    dpos[:, 0] = c * _DPOS
                if not _ALLPH:
                    dpos = dpos * grip.unsqueeze(-1); dhand = dhand * grip.unsqueeze(-1)
                act = ref.clone()
                act[:, 0:3] = act[:, 0:3] + dpos[src]
                act[:, env.hand_dof_start_idx:] = act[:, env.hand_dof_start_idx:] + dhand[src]
                env.step(act)
                if getattr(env, "_br_active_win", False) and env._br_win == 0:
                    p_ = env.root_state_tensor[env.object_indices, 0:3]
                    dd = torch.stack([torch.norm(p_[g * n:(g + 1) * n] - p_[:n], dim=-1)
                                      for g in range(2, G)], 0)
                    _, per = _reward(dd, torch)
                    gm = env._br_grip_mask()[:n]
                    for r in range(R):
                        m_ = gm & (rep == r)
                        if bool(m_.any()):
                            acc[r]["hold"].append(per[:, m_].mean(1).detach().cpu().numpy())
            sc = (env.successes[:n] > 0.5).float()
            for r in range(R):
                acc[r]["succ"].append(float(sc[rep == r].mean()))
        print("\n[res] === Δ 민감도 감사 (sweep=%s) ===" % _SWEEP, flush=True)
        print("[res] %8s %8s | %s | %8s %8s" % ("Δ수준", "liftSR", " ".join("%6s" % d for d in
              ["+x", "-x", "+y", "-y", "+z", "-z"]), "hold평균", "all6"), flush=True)
        for r in range(R):
            h = np.mean(np.stack(acc[r]["hold"]), 0)
            print("[res] %8.2f %8.3f | %s | %8.3f %8.3f"
                  % (float(lev[r]), float(np.mean(acc[r]["succ"])),
                     " ".join("%6.3f" % v for v in h), float(h.mean()), float(np.prod(h))), flush=True)
        raise SystemExit(0)

    if _EVAL:
        print("[res] EVAL — 원 eval 루프 10 라운드, residual 은 결정적(mean) 주입", flush=True)
        allenv = torch.arange(env.num_envs, device=dev)
        _DSTAT = []; _HOLD = []
        for rd in range(10):
            cur = env.reset_idx(allenv)["obs"]
            with torch.no_grad():
                plan = ppo_self.actor_critic(cur, env.get_state(), inference=True)
                if _PLAN:
                    dp = torch.tanh(net.pi(norm(priv_obs(env), update=False))) * _DPLAN
                    k = dp.shape[-1]
                    plan[:, :k] = torch.clamp(plan[:, :k] + dp[src], -1.0, 1.0)
            env.generate_reaching_plan_idx(allenv, actions=plan)
            for t in range(T):
                if _PLAN:                      # 계획만 바꾸고 스텝 행동은 원 레퍼런스 그대로
                    env.step(env.compute_reference_actions())
                    if getattr(env, "_br_active_win", False) and env._br_win == 0:
                        p_ = env.root_state_tensor[env.object_indices, 0:3]
                        dd = torch.stack([torch.norm(p_[g2 * n:(g2 + 1) * n] - p_[:n], dim=-1)
                                          for g2 in range(2, G)], 0)
                        _, per = _reward(dd, torch)
                        gm = env._br_grip_mask()[:n]
                        if bool(gm.any()):
                            _HOLD.append(per[:, gm].mean(1).detach().cpu().numpy())
                    continue
                ref = env.compute_reference_actions()
                on = norm(priv_obs(env), update=False)
                with torch.no_grad():
                    a = net.pi(on)
                grip = env._br_grip_mask()[:n].float()
                if _MODE == "zero":
                    a = a * 0
                th = torch.tanh(a)
                if bool((grip > 0).any()):
                    _DSTAT.append(th[grip > 0].detach())
                dpos = th[:, :3] * _DPOS * grip.unsqueeze(-1)
                dhand = th[:, 3:] * _DHAND * grip.unsqueeze(-1)
                act = ref.clone()
                act[:, 0:3] = act[:, 0:3] + dpos[src]
                act[:, env.hand_dof_start_idx:] = act[:, env.hand_dof_start_idx:] + dhand[src]
                env.step(act)
            if _DSTAT:
                dp = torch.cat(_DSTAT, 0)
                print("[res]   |Δ| 포화도: pos %.3f (클램프 대비), hand %.3f | mean|tanh| = %s"
                      % (dp[:, :3].abs().max(-1).values.mean(), dp[:, 3:].abs().max(-1).values.mean(),
                         " ".join("%.2f" % v for v in dp.abs().mean(0))), flush=True)
                _DSTAT.clear()
            print("[res] eval round %d  SR(main)=%.4f" % (rd, float((env.successes[:n] > 0.5).float().mean())), flush=True)
        if _HOLD:
            h = np.mean(np.stack(_HOLD), 0)
            print("[res] === hold/dir %s | 평균 %.4f | all6 %.4f ==="
                  % (" ".join("%s=%.3f" % (d, v) for d, v in
                     zip(["+x", "-x", "+y", "-y", "+z", "-z"], h)), h.mean(), np.prod(h)), flush=True)
        env.reset_idx(allenv)          # 마지막 라운드 successes 를 gs_branch 가 봉인하게 한다
        raise SystemExit(0)

    if _PLAN:
        dplan = 6 + env.num_active_hand_dofs
        pnet = Net(din, dplan).to(dev)
        popt = torch.optim.Adam(pnet.parameters(), lr=_LR)
        allenv = torch.arange(env.num_envs, device=dev)
        print("[res] PLAN 모드 — t=0 12-D edit 에 residual (dim=%d, ±%.2f), 집계=%s"
              % (dplan, _DPLAN, _AGG), flush=True)
        for it in range(_ITERS):
            cur = env.reset_idx(allenv)["obs"]
            o0 = norm(priv_obs(env), update=True)
            with torch.no_grad():
                teacher = ppo_self.actor_critic(cur, env.get_state(), inference=True)
                d = pnet.dist(o0)
                a0 = d.sample()
                lp0 = d.log_prob(a0).sum(-1)
                v0 = pnet.vf(o0).squeeze(-1)
            dp = torch.tanh(a0) * _DPLAN
            plan = teacher.clone()
            plan[:, :dplan] = torch.clamp(plan[:, :dplan] + dp[src], -1.0, 1.0)
            env.generate_reaching_plan_idx(allenv, actions=plan)
            Rsum = torch.zeros(n, device=dev); nw = 0; dm = []; hold = []
            for t in range(T):
                env.step(env.compute_reference_actions())
                if getattr(env, "_br_active_win", False) and env._br_win == 0:
                    p = env.root_state_tensor[env.object_indices, 0:3]
                    dd = torch.stack([torch.norm(p[g * n:(g + 1) * n] - p[:n], dim=-1)
                                      for g in range(2, G)], 0)
                    r, per = _reward(dd, torch)
                    Rsum = Rsum + r * env._br_grip_mask()[:n].float()
                    g_ = env._br_grip_mask()[:n]
                    if bool(g_.any()):
                        hold.append(per[:, g_].mean(1).detach().cpu().numpy())
                    nw += 1; dm.append(dd.max(0).values.detach().cpu().numpy())
            succ = (env.successes[:n] > 0.5).float()
            R = Rsum / max(nw, 1) + _WSUCC * succ
            adv = (R - v0); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(4):
                d = pnet.dist(o0); lp = d.log_prob(a0).sum(-1)
                ratio = (lp - lp0).exp()
                l = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean() \
                    + 0.5 * (pnet.vf(o0).squeeze(-1) - R).pow(2).mean() \
                    - 0.003 * d.entropy().sum(-1).mean()
                popt.zero_grad(); l.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(pnet.parameters(), 1.0); popt.step()
            dall = np.concatenate(dm) if dm else np.zeros(1)
            log.write("%d,%.4f,%.4f,%.4f,%.4f,%d,%.3f,%.3f,%.3f\n" % (
                it, float(R.mean()), float((Rsum / max(nw, 1)).mean()), 0.0, float(succ.mean()),
                nw, float(dall.max() * 1000), float(dall.mean() * 1000), float(dall.std() * 1000)))
            log.flush()
            if it % 5 == 0:
                h = np.mean(np.stack(hold), 0) if hold else np.zeros(6)
                print("[res]   hold/dir " + " ".join("%s=%.2f" % (d, v) for d, v in
                      zip(["+x", "-x", "+y", "-y", "+z", "-z"], h)), flush=True)
                print("[res] it=%3d R=%.3f margin=%.3f succ=%.3f win=%d dmean=%.1fmm"
                      % (it, float(R.mean()), float((Rsum / max(nw, 1)).mean()), float(succ.mean()),
                         nw, dall.mean() * 1000), flush=True)
            if it % 10 == 0 or it == _ITERS - 1:
                torch.save({"net": pnet.state_dict(), "norm_m": norm.m.cpu(), "norm_v": norm.v.cpu(),
                            "norm_n": norm.n, "mode": _MODE, "din": din, "dact": dplan},
                           os.path.join(out, "policy.pt"))
        log.close(); print("[res] done ->", out, flush=True); raise SystemExit(0)

    for it in range(_ITERS):
        obs_b, act_b, logp_b, val_b, rew_b, msk_b = [], [], [], [], [], []
        # ── 에피소드 1개 수집 (env 가 한 라운드 = 한 에피소드)
        cur = env.reset_idx(torch.arange(env.num_envs, device=dev))["obs"]
        with torch.no_grad():
            plan = ppo_self.actor_critic(cur, env.get_state(), inference=True)
        env.generate_reaching_plan_idx(torch.arange(env.num_envs, device=dev), actions=plan)
        stat = dict(rmin=[], rmean=[], nwin=0, d=[])
        for t in range(T):
            ref = env.compute_reference_actions()
            o = priv_obs(env)
            on = norm(o, update=True)
            with torch.no_grad():
                d = net.dist(on)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
                v = net.vf(on).squeeze(-1)
            grip = env._br_grip_mask()[:n].float()
            dpos = torch.tanh(a[:, :3]) * _DPOS * grip.unsqueeze(-1)
            dhand = torch.tanh(a[:, 3:]) * _DHAND * grip.unsqueeze(-1)
            if _MODE == "zero":
                dpos = dpos * 0; dhand = dhand * 0
            act = ref.clone()
            act[:, 0:3] = act[:, 0:3] + dpos[src]                       # 그룹 전체에 main 의 Δ
            act[:, env.hand_dof_start_idx:] = act[:, env.hand_dof_start_idx:] + dhand[src]
            env.step(act)

            r = torch.zeros(n, device=dev)
            if getattr(env, "_br_active_win", False) and env._br_win == 0:   # ★ 윈도 H 스텝 완료
                p = env.root_state_tensor[env.object_indices, 0:3]
                dd = torch.stack([torch.norm(p[g * n:(g + 1) * n] - p[:n], dim=-1) for g in range(2, G)], 0)
                r, per = _reward(dd, torch)
                r = r * grip
                stat["rmin"].append(per.min(0).values[grip > 0].mean().item() if (grip > 0).any() else 0.0)
                stat["rmean"].append(per.mean(0)[grip > 0].mean().item() if (grip > 0).any() else 0.0)
                if (grip > 0).any():
                    stat.setdefault("hold", []).append(per[:, grip > 0].mean(1).detach().cpu().numpy())
                stat["d"].append(dd.max(0).values[grip > 0].detach().cpu().numpy() if (grip > 0).any() else np.zeros(1))
                stat["nwin"] += 1
            if t == T - 2:
                r = r + _WSUCC * (env.successes[:n] > 0.5).float()
            obs_b.append(on); act_b.append(a); logp_b.append(lp); val_b.append(v); rew_b.append(r)
            msk_b.append(grip)

        succ = float((env.successes[:n] > 0.5).float().mean().item())
        R = torch.stack(rew_b)                    # [T, n]
        V = torch.stack(val_b)
        ret_mean = float(R.sum(0).mean().item())
        dall = np.concatenate(stat["d"]) if stat["d"] else np.zeros(1)
        log.write("%d,%.4f,%.4f,%.4f,%.4f,%d,%.3f,%.3f,%.3f\n" % (
            it, ret_mean, float(np.mean(stat["rmin"]) if stat["rmin"] else 0),
            float(np.mean(stat["rmean"]) if stat["rmean"] else 0), succ, stat["nwin"],
            float(dall.max() * 1000), float(dall.mean() * 1000), float(dall.std() * 1000)))
        log.flush()
        if it % 5 == 0:
            h = np.mean(np.stack(stat["hold"]), 0) if stat.get("hold") else np.zeros(6)
            print("[res]   hold/dir " + " ".join("%s=%.2f" % (d, v) for d, v in
                  zip(["+x", "-x", "+y", "-y", "+z", "-z"], h)), flush=True)
            print("[res] it=%3d ret=%.3f r_min=%.3f r_mean=%.3f succ=%.3f win=%d dmax=%.2fmm"
                  % (it, ret_mean, np.mean(stat["rmin"]) if stat["rmin"] else 0,
                     np.mean(stat["rmean"]) if stat["rmean"] else 0, succ, stat["nwin"],
                     dall.mean() * 1000), flush=True)
        # ── PPO 갱신 (GAE)
        with torch.no_grad():
            adv = torch.zeros_like(R); last = torch.zeros(n, device=dev)
            for t in reversed(range(T)):
                nv = V[t + 1] if t + 1 < T else torch.zeros(n, device=dev)
                delta = R[t] + 0.99 * nv - V[t]
                last = delta + 0.99 * 0.95 * last
                adv[t] = last
            ret = adv + V
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O = torch.stack(obs_b).reshape(-1, din); A = torch.stack(act_b).reshape(-1, dact)
        LP = torch.stack(logp_b).reshape(-1); AD = adv.reshape(-1); RT = ret.reshape(-1)
        M = torch.stack(msk_b).reshape(-1) > 0                 # grip 스텝만 학습
        if M.sum() < 32:
            continue
        O, A, LP, AD, RT = O[M], A[M], LP[M], AD[M], RT[M]
        idx = torch.randperm(O.shape[0], device=dev)
        for _ in range(4):
            for s in range(0, O.shape[0], max(256, O.shape[0] // 4)):
                b = idx[s:s + max(256, O.shape[0] // 4)]
                d = net.dist(O[b]); lp = d.log_prob(A[b]).sum(-1)
                ratio = (lp - LP[b]).exp()
                l_pi = -torch.min(ratio * AD[b], ratio.clamp(0.8, 1.2) * AD[b]).mean()
                l_v = (net.vf(O[b]).squeeze(-1) - RT[b]).pow(2).mean()
                loss = l_pi + 0.5 * l_v - 0.003 * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 10 == 0 or it == _ITERS - 1:
            _save_ck()
    _save_ck()
    log.close()
    print("[res] done ->", out, flush=True)
    raise SystemExit(0)


def install():
    if not _ON:
        return
    import importlib.abc, importlib.machinery, sys

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "algo.ppo_onestep.ppo":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None:
                return None
            inner = spec.loader

            class _L(importlib.abc.Loader):
                def create_module(s, sp):
                    return inner.create_module(sp)

                def exec_module(s, mod):
                    inner.exec_module(mod)
                    P = getattr(mod, "PPO", None)
                    if P is not None:
                        P.run = lambda self: _run(self)
                        print("[res] PPO.run hijacked (residual RL)", flush=True)

            spec.loader = _L()
            return spec

    sys.meta_path.insert(0, _Hook())
