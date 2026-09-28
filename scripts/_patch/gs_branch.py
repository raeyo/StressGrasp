"""graspstress 분기 마진 — grip phase 에서 ±xyz 6방향 외력 마진 M(s) 를 측정한다.

설계 정본 = docs/EXP_BRANCH.md (v2). 우산 README §1: 남의 코드(DemoGrasp)는 고치지 않는다.

■ 원리
DemoGrasp 는 env i 에 물체 `i % n_obj` 를 배정한다 (tasks/grasp.py:399). `num_envs = n_obj × G`
로 잡으면 env {j, j+n_obj, …} 는 같은 물체다. 이 세로줄이 한 **그룹**.
  g=0  main    : 기준. 외력 없음
  g=1  control : 복제본. 외력 없음  → 복제 충실도 게이트
  g≥2  probe   : 복제본 + (dir, α) 외력

■ ★ v2 에서 바뀐 것 — 에피소드 중 상태 복사를 하지 않는다
v1 은 grip 매 스텝마다 main 상태를 그림자에 복사했다. 실측 결과 **물체 root 를 수평으로
텔레포트하면 지지면 접촉이 깨져 물체가 테이블을 통과해 바닥으로 떨어졌다**
(정확히 18mm = tableHeight. 수직 1mm 이동·무동작 push 는 멀쩡했다 → 값 변경이 원인).
그래서 v2 는 **리셋 직후 물체가 공중에 있을 때 한 번만 복제**하고, 이후 물리는 건드리지 않는다.
복제본은 결정적 시뮬레이션에서 main 과 같은 궤적을 그리므로, 외력을 건 시점 이후의 차이가
곧 외력의 효과다. control 그룹이 main 과 일치하는지가 그 전제의 검증이다.

■ 판정
성공 여부는 DemoGrasp 원 판정식을 그대로 쓴다. M(s) = 6방향 전부 성공을 유지하는 최대 α.

환경변수:
  GS_BRANCH=1        켜기            GS_BR_ALPHA   α 사다리 (무게 배수)
  GS_BR_DOSE         외력 지속 env-step (기본 1)
  GS_BR_OFFSET       grip 진입 후 몇 스텝에 걸까. "auto"=라운드 번호 (기본), 정수면 고정
  GS_BR_SNAP         변위를 읽는 substep (기본 "1,2,5,10,20")
  GS_BR_DUMP         결과 npz 경로
"""
import os

_ON = os.environ.get("GS_BRANCH", "") not in ("", "0")
_ALPHAS = tuple(float(x) for x in os.environ.get("GS_BR_ALPHA", "1,2,5").split(",") if x.strip())
_SNAPS = tuple(int(x) for x in os.environ.get("GS_BR_SNAP", "1,2,5,10,20").split(",") if x.strip())
_DOSE = int(os.environ.get("GS_BR_DOSE", "1") or 1)
_OFFSET = os.environ.get("GS_BR_OFFSET", "auto")
_DUMP = os.environ.get("GS_BR_DUMP", "")
_BCAST_AT = int(os.environ.get("GS_BR_BCAST_AT", "2") or 2)  # reset 후 몇 번째 simulate 에서 복제
_RESYNC = os.environ.get("GS_BR_RESYNC", "") not in ("", "0")  # grip 중 주기적 재동기화 (RL 보상용)
_RE_EVERY = int(os.environ.get("GS_BR_RESYNC_EVERY", "1") or 1)  # 재동기화 주기(env-step). 분기 지평 H
_FRAME = os.environ.get("GS_BR_FRAME", "world")               # world | palm — 외력 방향의 기준 좌표계
_REPL = int(os.environ.get("GS_BR_REPLICAS", "1") or 1)       # 물체당 main env 복제 수 (RL 배치용)

_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _group_spec():
    spec = [(0.0, 0), (0.0, 0)]                      # 0=main, 1=control
    for a in _ALPHAS:
        for d in range(len(_DIRS)):
            spec.append((a, d))
    return spec


def _wrap(Grasp):
    if getattr(Grasp, "_br_patched", False):
        return
    from isaacgym import gymapi, gymtorch
    from isaacgym.torch_utils import quat_apply
    import numpy as np
    import torch

    orig_create_envs = Grasp._create_envs
    orig_reset = Grasp.reset_idx
    orig_plan = Grasp.generate_reaching_plan_idx
    orig_pre = Grasp.pre_physics_step
    SPEC = _group_spec()

    def _create_envs(self, *a, **kw):
        out = orig_create_envs(self, *a, **kw)
        gym, sim, dev, N = self.gym, self.sim, self.device, self.num_envs
        n_obj = len(getattr(self, "object_pcls", None) or getattr(self, "object_fns", [1]))
        n_main = n_obj * _REPL                      # main env 수 (물체당 _REPL 개 복제)
        G = N // n_main
        assert N % n_main == 0 and G == len(SPEC), \
            "num_envs 는 n_obj(%d) x replicas(%d) x G(%d) = %d 여야 한다 (지금 %d)" % (
                n_obj, _REPL, len(SPEC), n_main * len(SPEC), N)

        idx = torch.arange(N, device=dev)
        self._br_n_obj, self._br_n_main, self._br_G = n_obj, n_main, G
        self._br_src = idx % n_main                 # 각 env 의 main env 인덱스
        self._br_obj = idx % n_obj                  # 물체 id (DemoGrasp 가 i % n_obj 로 배정)
        self._br_group = idx // n_main
        self._br_sh = idx[self._br_group > 0]
        self._br_sh_src = self._br_src[self._br_sh]

        rb, ms = [], []
        for e in self.envs:
            h = gym.find_actor_handle(e, "object")
            rb.append(gym.get_actor_rigid_body_index(e, h, 0, gymapi.DOMAIN_SIM))
            ms.append(sum(p.mass for p in gym.get_actor_rigid_body_properties(e, h)))
        self._br_obj_rb = torch.tensor(rb, dtype=torch.long, device=dev)
        self._br_mass = torch.tensor(ms, dtype=torch.float, device=dev)
        self._br_force = torch.zeros((gym.get_sim_rigid_body_count(sim), 3), dtype=torch.float, device=dev)

        gl = self._br_group.tolist()
        self._br_alpha = torch.tensor([SPEC[g][0] for g in gl], dtype=torch.float, device=dev)
        self._br_dirvec = torch.tensor([_DIRS[SPEC[g][1]] for g in gl], dtype=torch.float, device=dev)
        self._br_dir_idx = torch.tensor([SPEC[g][1] for g in gl], dtype=torch.long, device=dev)

        self._br_round = 0      # ★ 0-based. 에피소드가 실제로 돈 경우에만 증가한다
        self._br_bcast = 0
        self._br_k = 0
        self._br_fire = torch.zeros(N, dtype=torch.bool, device=dev)
        self._br_t_grip = torch.full((n_main,), -1, dtype=torch.long, device=dev)
        self._br_fired = torch.zeros(n_main, dtype=torch.bool, device=dev)  # 외력이 실제로 발사됐나
        self._br_win, self._br_widx, self._br_active_win = 0, -1, False
        self._br_dirw = self._br_dirvec
        self._br_griplen = torch.zeros(n_main, dtype=torch.long, device=dev)
        self._br_applied = 0
        self._br_rec = {"round": [], "step": [], "k": [], "win": [], "widx": [],
                        "disp": [], "dz": [], "objz": [], "dtheta": []}
        self._br_succ, self._br_succ_round, self._br_tg = [], [], []
        self._br_fired_log, self._br_griplen_log = [], []
        self._br_bcast_err = []
        self.gym = _GymProxy(self.gym, self)
        import atexit
        def _final():
            try:
                if getattr(self, "_br_ep_ran", False) and (
                        not self._br_succ_round or self._br_succ_round[-1] != self._br_round):
                    _snap_round(self)       # ★ 마지막 라운드는 reset 이 안 오므로 여기서 봉인
            except Exception:
                pass
            _save(self)
        atexit.register(_final)
        print("[br] v3 | groups=%d (main + control + %d probe) | n_obj=%d x repl=%d = n_main=%d | envs=%d"
              " | alpha=%s dose=%d offset=%s H=%d frame=%s | obj_mass_mean=%.4fkg"
              % (G, G - 2, n_obj, _REPL, n_main, N, _ALPHAS, _DOSE, _OFFSET, _RE_EVERY, _FRAME,
                 self._br_mass.mean().item()), flush=True)
        return out

    # ---------------------------------------------------------- 복제 (리셋 직후 1회)
    def _br_broadcast(self):
        """main -> 그림자 복제. reset 직후(에피소드 첫 step 의 _BCAST_AT 번째 substep)에 1회.
        ★ root_state_tensor 는 env-local 이다 — env origin 을 더하면 안 된다 (계측기 결함 #6)."""
        gym, sim = self.gym, self.sim
        sh, src, rs, oi = self._br_sh, self._br_sh_src, self.root_state_tensor, self.object_indices
        if True:
            # ★ root_state_tensor 는 **env-local** 좌표다. env origin 을 더하면 물체가 이웃 env
            #   자리로 날아가 지지면(매트) 밖에 떨어진다 (실측: 정확히 18mm = 매트 높이만큼 낮게 안착).
            #   근거 = 원본 reset_idx 가 spawn 범위(x 0.3~0.8)를 전 env 에 그대로 쓴다.
            rs[oi[sh], 0:13] = rs[oi[src], 0:13]
            push = oi[sh].to(torch.int32)
            gym.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(rs), gymtorch.unwrap_tensor(push), push.numel())
        ds = self.dof_state.view(self.num_envs, -1, 2)
        ds[sh] = ds[src]
        ri = self.robot_indices[sh].to(torch.int32)
        gym.set_dof_state_tensor_indexed(
            sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(ri), ri.numel())
        self.prev_targets[sh] = self.prev_targets[src]
        self.cur_targets[sh] = self.cur_targets[src]
        gym.set_dof_position_target_tensor_indexed(
            sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(ri), ri.numel())

    def generate_reaching_plan_idx(self, env_ids, actions=None, *a, **kw):
        """계획·레퍼런스도 복제한다. randomizeTrackingReference/GraspPose 가 env 마다
        다른 난수를 뽑으므로, 이것을 맞추지 않으면 그림자가 다른 궤적을 따라간다."""
        out = orig_plan(self, env_ids, actions, *a, **kw)
        if not _ON:
            return out
        sh, src = self._br_sh, self._br_sh_src
        for key, t in list(self.current_tracking_reference.items()):
            if torch.is_tensor(t) and t.shape[0] == self.num_envs:
                if t is getattr(self, "tracking_reference", {}).get(key, None):
                    continue                       # 원본 공유 텐서면 env 별 차이가 없다
                t[sh] = t[src]
        for name in ("reaching_plan_ee", "reaching_plan_timesteps", "object_init_states"):
            t = getattr(self, name, None)
            if torch.is_tensor(t) and t.shape[0] == self.num_envs:
                t[sh] = t[src]
        return out

    def reset_idx(self, env_ids, *a, **kw):
        try:
            allenv = _ON and (len(env_ids) == self.num_envs)
        except TypeError:
            allenv = False
        if allenv and getattr(self, "_br_ep_ran", False):
            _snap_round(self)          # ★ 에피소드가 실제로 돈 경우에만 봉인 (빈 행 방지)
        if allenv and getattr(self, "_br_ep_ran", False):
            self._br_ep_done = True
        self._br_ep_ran = False
        out = orig_reset(self, env_ids, *a, **kw)
        if not _ON:
            return out
        if allenv and getattr(self, "_br_ep_done", False):
            self._br_round += 1          # ★ 돈 에피소드 수만 센다 (코드 경로별 여분 reset 무시)
        if allenv:
            self._br_ep_done = False
            self._br_t_grip.fill_(-1)
            self._br_fired.zero_()
            self._br_griplen.zero_()
            _save(self)
        self._br_bcast = _BCAST_AT                  # N번째 simulate 직전에 복제
        return out

    # ---------------------------------------------------------- 시점 · 외력
    def _br_grip_mask(self):
        """★ 성공 판정식(reward.py:reward_binary)이 쓰는 값을 그대로 재계산한다."""
        obj = self.object_pos
        dz = obj[:, 2] - self.object_init_states[:, 2]
        palm_d = torch.norm(obj - self.palm_center_pos, dim=-1)
        ft = torch.zeros_like(dz)
        for i in range(self.fingertip_pos.shape[-2]):
            ft += torch.norm(self.fingertip_pos[:, i, :] - obj, dim=-1)
        flag = (torch.clamp(ft, max=3.0) <= 0.12 * self.num_fingers) | (palm_d <= 0.15)
        return flag & (dz <= 0.1)

    def pre_physics_step(self, actions):
        out = orig_pre(self, actions)
        if not _ON:
            return out
        self._br_ep_ran = True
        n = self._br_n_main
        grip = self._br_grip_mask()[:n]                       # main 그룹이 시점을 정한다
        t = self.progress_buf[:n].long()
        fresh = grip & (self._br_t_grip < 0)
        self._br_t_grip = torch.where(fresh, t, self._br_t_grip)
        off = self._br_round if _OFFSET == "auto" else int(_OFFSET)
        age = t - self._br_t_grip
        on = (self._br_t_grip >= 0) & (age >= off) & (age < off + _DOSE) & grip
        self._br_fired |= on
        self._br_griplen += grip.long()
        if _RESYNC and bool(grip.any()):
            on = grip                               # 분기 지평 H 동안 계속 외력
            if self._br_win == 0:
                self._br_broadcast()                # H 스텝마다 상태를 main 으로 되돌린다
                self._br_widx += 1
            self._br_win = (self._br_win + 1) % max(1, _RE_EVERY)
            self._br_active_win = True              # ★ 이 스텝이 윈도 안이다
        elif _RESYNC:
            self._br_win = 0
            self._br_active_win = False
        f = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        f[self._br_sh] = on[self._br_sh_src]
        self._br_fire = f & (self._br_alpha > 0)
        self._br_on_any = bool(on.any())
        self._br_k = 0
        if _FRAME == "palm":
            # ★ 외력 방향을 손바닥 좌표계로 회전. main 의 palm_rot 을 그룹 전체에 쓴다
            #   (그림자는 복제본이라 같지만, 기준을 하나로 고정해 두는 편이 해석이 명확하다).
            self._br_dirw = quat_apply(self.palm_rot[self._br_src], self._br_dirvec)
        return out

    def _br_apply(self):
        """★ simulate() 직전마다 — apply_rigid_body_force_tensors 는 다음 simulate 1회에만 걸린다."""
        if not bool(self._br_fire.any()):
            return
        mag = self._br_alpha * self._br_mass * 9.81 * self._br_fire.float()
        dirs = self._br_dirw if _FRAME == "palm" else self._br_dirvec
        self._br_force.zero_()
        self._br_force[self._br_obj_rb] = dirs * mag.unsqueeze(-1)
        self._br_applied += int((mag > 0).sum().item())
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self._br_force), None, gymapi.ENV_SPACE)

    def _br_snap(self):
        if not getattr(self, "_br_on_any", False):
            return
        self._br_k += 1
        if self._br_k not in _SNAPS:
            return
        self.gym.refresh_actor_root_state_tensor(self.sim)
        rs, oi, src = self.root_state_tensor, self.object_indices, self._br_src
        p = rs[oi, 0:3]
        q = rs[oi, 3:7]
        dv = p - p[src]
        r = self._br_rec
        r["round"].append(self._br_round); r["step"].append(int(self.progress_buf[0].item())); r["k"].append(self._br_k)
        r["win"].append(self._br_win); r["widx"].append(self._br_widx)
        r["disp"].append(torch.norm(dv, dim=-1).cpu().numpy().copy())
        r["dz"].append(dv[:, 2].cpu().numpy().copy())
        r["objz"].append(p[:, 2].cpu().numpy().copy())
        r["dtheta"].append((2.0 * torch.acos((q * q[src]).sum(-1).abs().clamp(max=1.0))).cpu().numpy().copy())

    class _GymProxy(object):
        __slots__ = ("_g", "_t")

        def __init__(self, gym, task):
            object.__setattr__(self, "_g", gym); object.__setattr__(self, "_t", task)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_g"), name)

        def simulate(self, *a, **kw):
            t = object.__getattribute__(self, "_t")
            if t._br_bcast > 0:
                t._br_bcast -= 1
                if t._br_bcast == 0:
                    t._br_broadcast()
            t._br_apply()
            out = object.__getattribute__(self, "_g").simulate(*a, **kw)
            t._br_snap()
            return out

    def _snap_round(self):
        """한 라운드가 끝날 때(다음 reset 직전 또는 종료 시) 그 라운드의 결과를 봉인한다."""
        self._br_succ.append(self.successes.detach().cpu().numpy().copy())
        self._br_succ_round.append(self._br_round)
        self._br_tg.append(self._br_t_grip.detach().cpu().numpy().copy())
        self._br_fired_log.append(self._br_fired.detach().cpu().numpy().copy())
        self._br_griplen_log.append(self._br_griplen.detach().cpu().numpy().copy())

    def _save(self):
        if not _DUMP or not self._br_rec["round"]:
            return
        r = self._br_rec
        np.savez_compressed(
            _DUMP,
            round=np.array(r["round"], np.int32), step=np.array(r["step"], np.int32), k=np.array(r["k"], np.int32),
            win=np.array(r["win"], np.int32), widx=np.array(r["widx"], np.int32),
            disp=np.stack(r["disp"]).astype(np.float32), dz=np.stack(r["dz"]).astype(np.float32),
            objz=np.stack(r["objz"]).astype(np.float32), dtheta=np.stack(r["dtheta"]).astype(np.float32),
            success=np.stack(self._br_succ).astype(np.float32) if self._br_succ else np.zeros((0, self.num_envs), np.float32),
            success_round=np.array(self._br_succ_round, np.int32),
            t_grip=np.stack(self._br_tg).astype(np.int32) if self._br_tg else np.zeros((0, 1), np.int32),
            fired=np.stack(self._br_fired_log).astype(np.bool_) if self._br_fired_log else np.zeros((0, 1), np.bool_),
            grip_len=np.stack(self._br_griplen_log).astype(np.int32) if self._br_griplen_log else np.zeros((0, 1), np.int32),
            group=self._br_group.cpu().numpy().astype(np.int32), obj_id=self._br_obj.cpu().numpy().astype(np.int32),
            main_id=self._br_src.cpu().numpy().astype(np.int32),
            alpha=self._br_alpha.cpu().numpy().astype(np.float32), dir_idx=self._br_dir_idx.cpu().numpy().astype(np.int32),
            mass=self._br_mass.cpu().numpy().astype(np.float32),
            n_obj=self._br_n_obj, n_main=self._br_n_main, replicas=_REPL,
            G=self._br_G, dose=_DOSE, offset=str(_OFFSET),
            frame=_FRAME, resync=int(_RESYNC), resync_every=_RE_EVERY,
            snaps=np.array(_SNAPS, np.int32), alphas=np.array(_ALPHAS, np.float32), dirs=np.array(_DIRS, np.float32),
            force_applies=self._br_applied)
        print("[br] saved %s (%d events, rounds=%d, applies=%d)"
              % (_DUMP, len(r["round"]), len(self._br_succ), self._br_applied), flush=True)

    Grasp._create_envs = _create_envs
    Grasp._br_broadcast = _br_broadcast
    Grasp._br_grip_mask = _br_grip_mask
    Grasp._br_apply = _br_apply
    Grasp._br_snap = _br_snap
    Grasp.generate_reaching_plan_idx = generate_reaching_plan_idx
    Grasp.reset_idx = reset_idx
    Grasp.pre_physics_step = pre_physics_step
    Grasp._br_patched = True
    print("[br] Grasp patched (branch margin v2)", flush=True)


def install():
    if not _ON:
        return
    import importlib.abc, importlib.machinery, sys

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "tasks.grasp":
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
                    if getattr(mod, "Grasp", None) is not None:
                        _wrap(mod.Grasp)

            spec.loader = _L()
            return spec

    sys.meta_path.insert(0, _Hook())
