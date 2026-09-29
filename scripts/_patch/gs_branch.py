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

GS_BR_OFFSET=posthold — 파지 후(post-lift) 안정성 창 (EXP_POSTLIFT.md, Stability_Metrics §2~§5).
  원 성공조건(flag ∧ Δz>0.1)이 처음 참이 된 뒤 HOLD 스텝 무교란 유지를 통과하면(t0) 기준 상대자세를
  저장하고 probe 에 외력: RAMP substep 선형 증가 → LOAD 스텝 유지 → 제거 후 RECOVER 스텝 관찰.
  GS_BR_HOLD (기본 3 env-step=1s) / GS_BR_LOAD (3=1s) / GS_BR_RAMP (6 substep=0.1s) / GS_BR_RECOVER (3=1s).
  방향은 t0 의 main palm 회전으로 한 번 고정(follower load 금지). GS_BR_FRAME 무시. GS_BR_RESYNC 와 상호배타.
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
_HOLD = int(os.environ.get("GS_BR_HOLD", "3") or 3)           # posthold: lift 후 무교란 유지 env-step
_LOAD = int(os.environ.get("GS_BR_LOAD", "3") or 3)           # posthold: 외력 인가 env-step
_RAMP = int(os.environ.get("GS_BR_RAMP", "6") or 6)           # posthold: 선형 증가 substep 수
_RECOVER = int(os.environ.get("GS_BR_RECOVER", "3") or 3)     # posthold: 외력 제거 후 관찰 env-step
_POSTHOLD = _OFFSET == "posthold"
if _POSTHOLD and _RESYNC:
    raise AssertionError("GS_BR_OFFSET=posthold 과 GS_BR_RESYNC 는 동시 사용 금지")

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
    from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul, quat_rotate_inverse
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
        if _POSTHOLD:
            nan = float("nan")
            self._ph_tlift = torch.full((n_main,), -1, dtype=torch.long, device=dev)
            self._ph_t0m = torch.full((n_main,), -1, dtype=torch.long, device=dev)   # main 별 t0
            self._ph_dead = torch.zeros(n_main, dtype=torch.bool, device=dev)
            self._ph_g0 = torch.zeros(n_main, dtype=torch.bool, device=dev)          # 초기 파지 성공
            self._ph_t0 = torch.full((N,), -1, dtype=torch.long, device=dev)         # env 별 t0(=main 것)
            self._ph_p0 = torch.zeros((N, 3), dtype=torch.float, device=dev)         # 기준 p_PO
            self._ph_q0 = torch.zeros((N, 4), dtype=torch.float, device=dev)         # 기준 q_PO
            self._ph_dirw = self._br_dirvec.clone()          # t0 palm 회전으로 고정된 외력 방향
            self._ph_done = torch.zeros(N, dtype=torch.bool, device=dev)
            self._ph_ep_max = torch.zeros(N, dtype=torch.float, device=dev)
            self._ph_eR_max = torch.zeros(N, dtype=torch.float, device=dev)
            self._ph_ep_load = torch.full((N,), nan, dtype=torch.float, device=dev)
            self._ph_eR_load = torch.full((N,), nan, dtype=torch.float, device=dev)
            self._ph_ep_end = torch.full((N,), nan, dtype=torch.float, device=dev)
            self._ph_eR_end = torch.full((N,), nan, dtype=torch.float, device=dev)
            self._ph_dz_end = torch.full((N,), nan, dtype=torch.float, device=dev)
            self._ph_ksub = 0                                # 현 env-step 내 simulate 호출 수(1부터)
            self._ph_dec = 20                                # ★ decimation: 관측으로 자기보정(아래)
            self._ph_logs = {k: [] for k in (
                "g0", "tlift", "t0", "done", "ep_load", "eR_load",
                "ep_end", "eR_end", "ep_max", "eR_max", "dz_end")}
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
        if _POSTHOLD:
            print("[br] posthold | hold=%d load=%d ramp=%d recover=%d (env-step/substep),"
                  " 방향=t0 palm 고정, resync 금지" % (_HOLD, _LOAD, _RAMP, _RECOVER), flush=True)
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
        # ★ plan_anchor_pos = 로컬 external/DemoGrasp 수정본에만 있다 (reset 마다 env 자기 물체 위치로
        #   재설정되고 레퍼런스 추적의 기준점이 된다). 복사 안 하면 1라운드부터 그림자가 물체를 놓친다
        #   (실측 2026-09-29: control SR 0.76 → 0.03~0.09). 원본에는 없으므로 getattr 로 건너뛴다.
        for name in ("reaching_plan_ee", "reaching_plan_timesteps", "object_init_states", "plan_anchor_pos"):
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
            if _POSTHOLD:
                self._br_ph_reset()
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

    # ---------------------------------------------------------- posthold (파지 후 안정성)
    def _ph_lifted_mask(self):
        """DemoGrasp 성공조건과 동일: flag ∧ Δz>0.1 (reward.py:reward_binary). _br_grip_mask 의 역방향."""
        obj = self.object_pos
        dz = obj[:, 2] - self.object_init_states[:, 2]
        palm_d = torch.norm(obj - self.palm_center_pos, dim=-1)
        ft = torch.zeros_like(dz)
        for i in range(self.fingertip_pos.shape[-2]):
            ft += torch.norm(self.fingertip_pos[:, i, :] - obj, dim=-1)
        flag = (torch.clamp(ft, max=3.0) <= 0.12 * self.num_fingers) | (palm_d <= 0.15)
        return flag & (dz > 0.1)

    def _br_ph_reset(self):
        """에피소드(라운드) 시작 시 posthold 상태 초기화."""
        self._ph_tlift.fill_(-1)
        self._ph_t0m.fill_(-1)
        self._ph_dead.zero_()
        self._ph_g0.zero_()
        self._ph_t0.fill_(-1)
        self._ph_p0.zero_()
        self._ph_q0.zero_()
        self._ph_dirw = self._br_dirvec.clone()
        self._ph_done.zero_()
        self._ph_ep_max.zero_()
        self._ph_eR_max.zero_()
        nan = float("nan")
        for k in ("ep_load", "eR_load", "ep_end", "eR_end", "dz_end"):
            getattr(self, "_ph_" + k).fill_(nan)

    def _br_ph_rebroadcast(self, sel_sh):
        """posthold t0: sel_sh(그림자 부분집합 마스크)에 대해서만 main -> 그림자 복제 (_br_broadcast 의 부분판).
        다른 main 의 창 도중인 그림자는 건드리지 않는다."""
        gym, sim = self.gym, self.sim
        sh, src = self._br_sh[sel_sh], self._br_sh_src[sel_sh]
        rs, oi = self.root_state_tensor, self.object_indices
        rs[oi[sh], 0:13] = rs[oi[src], 0:13]          # env-local — origin 보정 금지 (결함 #6)
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
        self._ph_nrebc = getattr(self, "_ph_nrebc", 0) + int(sh.numel())

    def _br_ph_step(self):
        """posthold 매 env-step(pre_physics_step) 본문. 시점은 main 이 정하고 그림자에 전파.

        ★ 관측 신선도: palm_center_pos/palm_rot/object_pos 는 DemoGrasp 가 post_physics_step 에서
          계산해 둔 캐시 값이다. pre_physics_step 직전까지 simulate 가 없으므로 이 값들이 '방금 끝난
          env-step 종료 시점' 의 최신 관측이다(기존 t_grip 기록·발사 시점 판정도 같은 관례 — palm 을
          다시 계산하는 것은 DemoGrasp 내부식을 복제하는 길이라 피한다). 물체 p·q 만
          refresh_actor_root_state_tensor 를 다시 받아 **같은 순간의 쌍**으로 읽는다(reset 직후 복제
          등 시점 예외에도 안전)."""
        m = self._br_n_main
        lifted = self._ph_lifted_mask()
        t = self.progress_buf.long()
        lifted_m, t_m = lifted[:m], t[:m]
        fresh = lifted_m & (self._ph_tlift < 0)
        self._ph_tlift = torch.where(fresh, t_m, self._ph_tlift)
        if bool(fresh.any()):
            # ★ t_lift 재복제 (Stability_Metrics §2.2 "t0 상태로 복원"의 근사). reset 직후 1회 복제만으로는
            #   파지 형성 중 그림자가 ~7% 갈라진다(복제 잡음 바닥, 실측). t0 에 바로 복제하면 텔레포트 직후
            #   PhysX 접촉 캐시 충격으로 control 물체가 15~176° 튄다(실측 12%) → lift 순간 복제하고
            #   HOLD(1 s) 동안 안정화시킨 뒤 t0 에서 각 env 자기 기준을 잡는다. 물체는 공중·손 안이라
            #   지지면 텔레포트 문제(결함 #6 조사)와 무관하다.
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self._br_ph_rebroadcast(fresh[self._br_sh_src])
        age = t_m - self._ph_tlift
        active = (self._ph_tlift >= 0) & (self._ph_t0m < 0) & ~self._ph_dead
        dead = active & (age >= 1) & ~lifted_m       # HOLD 중 lifted 깨짐 → g0=0 확정, 발사 없음
        self._ph_dead |= dead
        start = active & ~dead & (age >= _HOLD)      # t0 = t_lift + HOLD 에서 START
        self._ph_t0m = torch.where(start, t_m, self._ph_t0m)
        self._ph_g0 |= start                         # START 시 g0=1
        self._br_fired |= start                      # fired = START 도달해(외력이 걸린) main
        new_t0 = start[self._br_src]                 # 그림자는 main 의 t0 를 물려받는다
        self._ph_t0 = torch.where(new_t0, t, self._ph_t0)
        # ★ decimation 자기보정: 직전 env-step 에 관측된 simulate 호출 수가 곧 substep 수다
        #   (초깃값 20 = gs_stress.py 실측. t0 이전에 반드시 1회 이상 관측된다)
        if self._ph_ksub > 0:
            self._ph_dec = int(self._ph_ksub)
        self._ph_ksub = 0
        self._br_k = 0                               # _br_snap 이 substep 인덱스를 쓸 수 있게

        age_w = t - self._ph_t0
        meas = (self._ph_t0 >= 0) & (age_w >= 0) & (age_w <= _LOAD + _RECOVER)
        fire = (self._ph_t0 >= 0) & (age_w >= 0) & (age_w < _LOAD) & (self._br_alpha > 0)
        self._br_fire = fire
        self._br_on_any = bool(fire.any())

        if bool(meas.any()):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            rs, oi = self.root_state_tensor, self.object_indices
            obj_p = rs[oi, 0:3]
            obj_q = rs[oi, 3:7]
            pp, pq = self.palm_center_pos, self.palm_rot
            p_PO = quat_rotate_inverse(pq, obj_p - pp)
            q_PO = quat_mul(quat_conjugate(pq), obj_q)
            if bool(start.any()):
                # 기준 상대자세 = 각 env 자기 값 (t_lift 재복제 후 HOLD 동안 안정화된 상태)
                #   + 외력 방향을 t0 main palm 회전으로 한 번 고정
                n0 = new_t0.unsqueeze(-1)
                self._ph_p0 = torch.where(n0, p_PO, self._ph_p0)
                self._ph_q0 = torch.where(n0, q_PO, self._ph_q0)
                dirw = quat_apply(pq[self._br_src], self._br_dirvec)
                self._ph_dirw = torch.where(n0, dirw, self._ph_dirw)
            e_p = torch.norm(p_PO - self._ph_p0, dim=-1)
            e_R = 2.0 * torch.acos((q_PO * self._ph_q0).sum(-1).abs().clamp(max=1.0))
            self._ph_ep_max = torch.where(meas, torch.maximum(self._ph_ep_max, e_p), self._ph_ep_max)
            self._ph_eR_max = torch.where(meas, torch.maximum(self._ph_eR_max, e_R), self._ph_eR_max)
            at_load = meas & (age_w == _LOAD)
            self._ph_ep_load = torch.where(at_load, e_p, self._ph_ep_load)
            self._ph_eR_load = torch.where(at_load, e_R, self._ph_eR_load)
            at_end = meas & (age_w == _LOAD + _RECOVER)
            self._ph_ep_end = torch.where(at_end, e_p, self._ph_ep_end)
            self._ph_eR_end = torch.where(at_end, e_R, self._ph_eR_end)
            dz = obj_p[:, 2] - self.object_init_states[:, 2]
            self._ph_dz_end = torch.where(at_end, dz, self._ph_dz_end)
            self._ph_done |= at_end                  # 창 완료. 중도 종료면 done=0 으로 남는다

    def pre_physics_step(self, actions):
        out = orig_pre(self, actions)
        if not _ON:
            return out
        self._br_ep_ran = True
        if _POSTHOLD:
            self._br_ph_step()
            return out
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
        if _POSTHOLD:
            self._ph_ksub += 1                       # 현 env-step 의 몇 번째 substep 인지(1부터)
            if not bool(self._br_fire.any()):
                return
            # ramp: s = 부하 시작 후 몇 번째 substep(1부터). 크기 = α·m·g·min(1, s/RAMP)
            s = (self.progress_buf.long() - self._ph_t0) * self._ph_dec + self._ph_ksub
            if _RAMP > 0:
                ramp = torch.clamp(s.float() / float(_RAMP), max=1.0)
            else:
                ramp = torch.ones_like(s, dtype=torch.float)
            mag = self._br_alpha * self._br_mass * 9.81 * self._br_fire.float() * ramp
            self._br_force.zero_()
            self._br_force[self._br_obj_rb] = self._ph_dirw * mag.unsqueeze(-1)
            self._br_applied += int((mag > 0).sum().item())
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self._br_force), None, gymapi.ENV_SPACE)
            return
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
        if _POSTHOLD:
            nan = float("nan")
            started = self._ph_t0 >= 0
            L = self._ph_logs
            L["g0"].append(self._ph_g0.cpu().numpy().astype(np.float32).copy())
            L["tlift"].append(self._ph_tlift.cpu().numpy().astype(np.int32).copy())
            L["t0"].append(self._ph_t0m.cpu().numpy().astype(np.int32).copy())
            L["done"].append(torch.where(started, self._ph_done.float(),
                                         torch.full_like(self._ph_done.float(), nan)).cpu().numpy().copy())
            L["ep_max"].append(torch.where(started, self._ph_ep_max,
                                           torch.full_like(self._ph_ep_max, nan)).cpu().numpy().copy())
            L["eR_max"].append(torch.where(started, self._ph_eR_max,
                                           torch.full_like(self._ph_eR_max, nan)).cpu().numpy().copy())
            L["ep_load"].append(self._ph_ep_load.cpu().numpy().copy())
            L["eR_load"].append(self._ph_eR_load.cpu().numpy().copy())
            L["ep_end"].append(self._ph_ep_end.cpu().numpy().copy())
            L["eR_end"].append(self._ph_eR_end.cpu().numpy().copy())
            L["dz_end"].append(self._ph_dz_end.cpu().numpy().copy())

    def _br_ph_savedict(self):
        """posthold 결과를 _save 에 넘길 추가 키. 미정의(창 미시작)는 nan."""
        L = self._ph_logs
        R = len(L["g0"])
        n, m = self.num_envs, self._br_n_main

        def col(k, rows):
            return np.stack(L[k]).astype(np.float32) if R else np.zeros((0, rows), np.float32)

        return dict(
            ph_g0=col("g0", m),
            ph_tlift=col("tlift", m).astype(np.int32), ph_t0=col("t0", m).astype(np.int32),
            ph_done=col("done", n), ph_ep_load=col("ep_load", n), ph_eR_load=col("eR_load", n),
            ph_ep_end=col("ep_end", n), ph_eR_end=col("eR_end", n),
            ph_ep_max=col("ep_max", n), ph_eR_max=col("eR_max", n), ph_dz_end=col("dz_end", n),
            ph_hold=_HOLD, ph_load=_LOAD, ph_ramp=_RAMP, ph_recover=_RECOVER)

    def _save(self):
        if not _DUMP or not self._br_rec["round"]:
            return      # posthold 도 외력 이벤트(_br_snap)가 생긴 뒤에만 저장 — 빈 리스트 np.stack 방지
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
            force_applies=self._br_applied,
            **(_br_ph_savedict(self) if _POSTHOLD else {}))
        print("[br] saved %s (%d events, rounds=%d, applies=%d)"
              % (_DUMP, len(r["round"]), len(self._br_succ), self._br_applied), flush=True)

    Grasp._create_envs = _create_envs
    Grasp._br_broadcast = _br_broadcast
    Grasp._br_grip_mask = _br_grip_mask
    Grasp._br_apply = _br_apply
    Grasp._br_snap = _br_snap
    Grasp._ph_lifted_mask = _ph_lifted_mask
    Grasp._br_ph_reset = _br_ph_reset
    Grasp._br_ph_step = _br_ph_step
    Grasp._br_ph_rebroadcast = _br_ph_rebroadcast
    Grasp._br_ph_savedict = _br_ph_savedict
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
