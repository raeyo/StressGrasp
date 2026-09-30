"""POSTRES — 폐합 후(post-closure) 상수 Δ 의 **가동범위** 측정.  설계 = docs/EXP_POSTRES.md.

■ 무엇을 하나
  계획(12-D plan)은 전 슬롯 teacher 로 고정하고, **grip 진입 + K 스텝 이후부터 에피소드 끝까지**
  슬롯마다 다른 상수 Δ 를 레퍼런스 행동 위에 얹는다. 라벨은 gs_branch 의 posthold 창(α=8) 을 그대로 쓴다.

■ act 레이아웃 (arm_controller=pose, DemoGrasp tasks/grasp.py:1380)
      act[:, 0:3]   절대 EE 위치 (m)
      act[:, 3:7]   절대 EE 쿼터니언          ← ★ 선행 residual 이 한 번도 안 건드린 축
      act[:, 7:]    손 관절 목표 ([-1,1])      (hand_dof_start_idx = 7)
  Δ 는 **손(EE) 국소 좌표계**에서 준다:  p += R(q)·dp ,  q := q ⊗ dq(축각) ,  h += dh

■ 팔(arm):  0=ZERO  1=OLD(위치+손)  2=ROT(회전)  3=NEW(회전+손)      레벨: 0=L1 1=L2
  슬롯 0 = ZERO(대조군). 나머지 = 팔 3 × 레벨 2 × 후보 GS_PR_K 개.

환경변수: GS_POSTRES=1  GS_PR_EPS  GS_PR_K  GS_PR_KCAP  GS_PR_REPS  GS_PR_OUT
          GS_PR_LEVELS="dpos,drot,dhand;..." (레벨을 ; 로 구분. dpos m · drot rad · dhand 정규화)
"""
import os

_ON = os.environ.get("GS_POSTRES", "") not in ("", "0")
_EPS = int(os.environ.get("GS_PR_EPS", "20") or 20)
_K = int(os.environ.get("GS_PR_K", "4") or 4)
_KCAP = int(os.environ.get("GS_PR_KCAP", "5") or 5)
_REPS = int(os.environ.get("GS_PR_REPS", "2") or 2)
_OUT = os.environ.get("GS_PR_OUT", "")
_LV = [[float(x) for x in g.split(",")] for g in
       os.environ.get("GS_PR_LEVELS", "0.01,0.05,0.15;0.02,0.10,0.25;0.03,0.20,0.40").split(";") if g.strip()]
_NLEV = len(_LV)
# 팔 정의: (이름, 위치쓰나, 회전쓰나, 손쓰나)
_ARMS = [("OLD", 1, 0, 1), ("ROT", 0, 1, 0), ("NEW", 0, 1, 1)]
# posthold 판정 임계 (gs_residual dataset-posthold 와 동일 — 라벨 정의를 바꾸지 않는다)
_EP_END_MAX, _DZ_END_MIN, _EP_MAX_MAX, _ER_MAX_DEG = 0.05, 0.05, 0.02, 15.0


def _run(ppo_self):
    import time as _time
    import numpy as np
    import torch
    from isaacgym.torch_utils import quat_apply, quat_mul, quat_from_angle_axis

    env = ppo_self.vec_env
    dev = env.device
    n = env._br_n_main
    n_obj = env._br_n_obj
    R = n // n_obj
    G = env._br_G
    src = env._br_src
    T = int(env.max_episode_length)
    nh = int(env.num_active_hand_dofs)
    dp = 6 + nh
    hs = int(env.hand_dof_start_idx)
    out = _OUT or os.path.join(os.environ.get("GS_ROOT", "."), "runs", "postres", "pr")
    os.makedirs(out, exist_ok=True)
    allenv = torch.arange(env.num_envs, device=dev)
    ar = torch.arange(n, device=dev)
    slot = (ar // n_obj)                                   # [n] 0..R-1
    probe_idx = torch.stack([(2 + d) * n + ar for d in range(6)], 0)   # [6, n]
    ctrl_idx = n + ar

    # ---- 슬롯 -> (팔, 레벨) 배정
    n_need = 1 + len(_ARMS) * _NLEV * _K
    assert R >= n_need, "R(%d) < 필요 슬롯(%d). GS_REPL 을 %d 로." % (R, n_need, n_need)
    arm_of = np.zeros(R, dtype=np.int32)                   # 0 = ZERO
    lev_of = np.zeros(R, dtype=np.int32)
    s = 1
    for ai in range(len(_ARMS)):
        for li in range(_NLEV):
            for _ in range(_K):
                if s < R:
                    arm_of[s], lev_of[s] = ai + 1, li
                    s += 1
    arm_t = torch.tensor(arm_of, device=dev)[slot]          # [n]
    lev_t = torch.tensor(lev_of, device=dev)[slot]
    use = torch.tensor([[0., 0., 0.]] + [[float(a[1]), float(a[2]), float(a[3])] for a in _ARMS],
                       device=dev)[arm_t]                   # [n, 3] (pos, rot, hand)
    sc = torch.tensor(_LV, device=dev)[lev_t]               # [n, 3] 레벨별 크기

    print("[pr] POSTRES n_obj=%d R=%d G=%d n_env=%d eps=%d K=%d kcap=%d reps=%d nh=%d hs=%d T=%d"
          % (n_obj, R, G, env.num_envs, _EPS, _K, _KCAP, _REPS, nh, hs, T), flush=True)
    print("[pr] 슬롯배정 arm=%s lev=%s  levels=%s"
          % (list(arm_of), list(lev_of), _LV), flush=True)

    ST = {k: [] for k in ("obj", "slot", "arm", "lev", "scene", "dpos", "drot", "dhand")}
    LB = [{k: [] for k in ("g0", "yhold", "ypose", "done", "ctrl_hold", "succ", "on_frac")}
          for _ in range(_REPS)]
    P0 = []
    t_start = _time.time()

    for ep in range(_EPS):
        env.reset_idx(allenv)
        p0 = env.root_state_tensor[env.object_indices, 0:7].clone()
        p0 = p0[:n_obj].repeat(int(np.ceil(env.num_envs / n_obj)), 1)[:env.num_envs]
        p0np = p0.cpu().numpy()
        cur = env.reset_idx(allenv, object_init_pose=p0np)["obs"]
        with torch.no_grad():
            teach = ppo_self.actor_critic(cur, env.get_state(), inference=True)[:, :dp].clone()
        plan_main = teach[:n_obj].repeat(R, 1)              # ★ 전 슬롯 동일 = teacher 계획

        # ---- Δ 추첨 (구면 위. 장면×물체×슬롯마다 1회, A/B 공통)
        # ★ 크기 규약 (EXP_POSTRES §2 의 레벨을 그대로 읽는다):
        #   위치·손 = **좌표당 RMS** 가 레벨과 같도록 구면 반지름 = level·sqrt(dim)
        #     (선행 residual 의 클램프는 좌표별 박스 ±level 이었다. 구면 반지름을 level 로 두면
        #      18-DoF 손에서 좌표당 .035 밖에 안 되어 선행보다 훨씬 약한 Δ 가 된다.)
        #   회전 = **총 회전각(rad)** 이 레벨과 같도록 반지름 = level (축각 3-벡터의 노름 = 각도)
        def _sph(k, r):
            v = torch.randn(n, k, device=dev)
            return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8) * r
        import math as _math
        dpos = _sph(3, sc[:, 0:1] * use[:, 0:1] * _math.sqrt(3.0))
        drot = _sph(3, sc[:, 1:2] * use[:, 1:2])
        dhand = _sph(nh, sc[:, 2:3] * use[:, 2:3] * _math.sqrt(float(nh)))
        ang = drot.norm(dim=-1)
        axis = drot / ang.clamp(min=1e-8).unsqueeze(-1)
        dq = quat_from_angle_axis(ang, axis)                # [n,4]; ang=0 -> identity

        for rep in range(_REPS):
            env.reset_idx(allenv, object_init_pose=p0np)
            plan = torch.zeros(env.num_envs, dp, device=dev)
            plan[:n] = plan_main
            env.generate_reaching_plan_idx(allenv, actions=plan[src])

            N_ = env.num_envs
            t_grip = torch.full((n,), -1, dtype=torch.long, device=dev)
            g0 = torch.zeros(n, dtype=torch.bool, device=dev)
            mydone = torch.zeros(N_, dtype=torch.bool, device=dev)
            nan = float("nan")
            ep_end = torch.full((N_,), nan, device=dev); dz_end = torch.full((N_,), nan, device=dev)
            ep_max = torch.full((N_,), nan, device=dev); eR_max = torch.full((N_,), nan, device=dev)
            n_on = torch.zeros(n, device=dev)

            for t in range(T):
                ref = env.compute_reference_actions()
                grip = env._br_grip_mask()[:n]
                tt = env.progress_buf[:n].long()
                t_grip = torch.where(grip & (t_grip < 0), tt, t_grip)
                on = (t_grip >= 0) & ((tt - t_grip) >= _KCAP)        # 폐합 후부터 끝까지
                m = on.float().unsqueeze(-1)
                n_on += on.float()

                act = ref.clone()
                q = act[:, 3:7]
                act[:, 0:3] = act[:, 0:3] + quat_apply(q, (dpos * m)[src])
                # 국소 회전: q := q ⊗ dq  (Δ 가 꺼진 env 는 dq=identity 로 섞는다)
                w = m[src]
                dq_on = torch.cat([dq[src][:, 0:3] * w,
                                   torch.where(w > 0.5, dq[src][:, 3:4], torch.ones_like(w))], -1)
                dq_on = dq_on / dq_on.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                act[:, 3:7] = quat_mul(q, dq_on)
                act[:, hs:hs + nh] = act[:, hs:hs + nh] + (dhand * m)[src]
                env.step(act)

                g0 |= env._ph_g0[:n]
                nd = env._ph_done & (~mydone)
                if bool(nd.any()):
                    ep_end[nd] = env._ph_ep_end[nd]; dz_end[nd] = env._ph_dz_end[nd]
                    ep_max[nd] = env._ph_ep_max[nd]; eR_max[nd] = env._ph_eR_max[nd]
                    mydone |= nd

            yh = mydone & (ep_end <= _EP_END_MAX) & (dz_end > _DZ_END_MIN)
            yp = yh & (ep_max <= _EP_MAX_MAX) & (eR_max <= float(np.deg2rad(_ER_MAX_DEG)))
            L = LB[rep]
            L["g0"].append(g0.float().cpu().numpy())
            L["yhold"].append(yh[probe_idx].float().T.cpu().numpy())
            L["ypose"].append(yp[probe_idx].float().T.cpu().numpy())
            L["done"].append(mydone[probe_idx].float().T.cpu().numpy())
            L["ctrl_hold"].append(yh[ctrl_idx].float().cpu().numpy())
            L["succ"].append((env.successes[:n] > 0.5).float().cpu().numpy())
            L["on_frac"].append((n_on / float(T)).cpu().numpy())

        ST["obj"].append(env._br_obj[:n].cpu().numpy().astype(np.int32))
        ST["slot"].append(slot.cpu().numpy().astype(np.int32))
        ST["arm"].append(arm_t.cpu().numpy().astype(np.int32))
        ST["lev"].append(lev_t.cpu().numpy().astype(np.int32))
        ST["scene"].append(np.full(n, ep, dtype=np.int32))
        ST["dpos"].append(dpos.cpu().numpy().astype(np.float32))
        ST["drot"].append(drot.cpu().numpy().astype(np.float32))
        ST["dhand"].append(dhand.cpu().numpy().astype(np.float32))
        P0.append(p0np[:n_obj].copy())

        el = _time.time() - t_start
        a0 = LB[0]
        d_ = a0["done"][-1] > 0
        z = slot.cpu().numpy() == 0
        ap = (a0["g0"][-1] > 0) & d_.all(1) & (a0["ypose"][-1] > 0.5).all(1)
        print("[pr] ep %3d/%d  g0=%.3f done6=%.3f ctrlHold=%.3f onFrac=%.2f | R_all_pose: 슬롯0 %.3f / 전체 %.3f | %.0fs (%.1fs/ep, ETA %.0fm)"
              % (ep + 1, _EPS, float(a0["g0"][-1].mean()), float(d_.all(1).mean()),
                 float(np.nanmean(a0["ctrl_hold"][-1][a0["g0"][-1] > 0])) if (a0["g0"][-1] > 0).any() else float("nan"),
                 float(a0["on_frac"][-1].mean()), float(ap[z].mean()), float(ap.mean()),
                 el, el / (ep + 1), (_EPS - ep - 1) * el / (ep + 1) / 60.0), flush=True)

        d = {k: np.concatenate(v, 0) for k, v in ST.items()}
        d.update(p0_obj=np.stack(P0, 0), meta_R=R, meta_nobj=n_obj, meta_K=_K, meta_kcap=_KCAP,
                 meta_alpha=float(os.environ.get("GS_BR_ALPHA", "8")),
                 meta_arm_of=arm_of, meta_lev_of=lev_of, meta_L=np.array(_LV, dtype=np.float32),
                 meta_armnames=np.array(["ZERO"] + [a[0] for a in _ARMS]))
        for r in range(_REPS):
            for k, v in LB[r].items():
                d["%s_%s" % (k, "ABCDE"[r])] = np.concatenate(v, 0)
        np.savez_compressed(os.path.join(out, "data.npz"), **d)

    print("[pr] saved", os.path.join(out, "data.npz"), flush=True)
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
                        print("[pr] PPO.run hijacked (POSTRES)", flush=True)

            spec.loader = _L()
            return spec

    sys.meta_path.insert(0, _Hook())
