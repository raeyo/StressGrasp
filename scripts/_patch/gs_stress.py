"""graspstress 런타임 패치 — DemoGrasp(남의 코드)를 고치지 않고 스트레스 축을 주입한다.

우산 README §1: external/references 의 남의 코드는 수정하지 않는다. 패치는 우리 프로젝트에 두고
런타임에 얹는다. 이 모듈은 `tasks.grasp` 가 import 되는 순간 `Grasp` 를 래핑한다.

환경변수 (전부 미설정이면 무동작 = stock 동작):
  GS_MASS_SCALE : object 질량 배율 (결정적. 예: 4.0). D2 하중 축.
  GS_EXT_G      : 파지 후 물체에 가하는 외력, 물체 무게의 배수 (예: 2.0). D3 운동 축.
  GS_FORCE_MODE : 'planar'(기본, 에피소드마다 고정된 수평 방향) | 'iso'(매 스텝 등방 랜덤)
  GS_DUMP       : per-env 성공/물체 id 를 저장할 npz 경로.

★ 외력은 **성공 판정과 동일한 조건**(object_delta_z > 0.1)에서만 발화한다.
  판정식이 쓰는 값을 그대로 써서 "언제부터 post-lift 인가"의 정의 불일치를 원천 차단한다.
"""
import os

# ★ 여기서 numpy/torch 를 import 하지 않는다. 이 모듈은 usercustomize 로 인터프리터 시작 시
#   로드되는데, torch 가 isaacgym 보다 먼저 올라가면 isaacgym 이 ImportError 로 죽는다
#   (MACHINES.md §0). numpy/torch 는 _wrap() 안에서 — 즉 isaacgym 로드 이후에만 — 당긴다.

_MASS = float(os.environ.get("GS_MASS_SCALE", "1") or 1)
_EXTG = float(os.environ.get("GS_EXT_G", "0") or 0)
_MODE = os.environ.get("GS_FORCE_MODE", "planar")
_PHASE = os.environ.get("GS_PHASE", "post")   # pre | grasp | post | all — 외란을 거는 시점
# ★ 투여량(dose) 통제: 시점 진입 후 몇 **env-step** 동안 힘을 거는가. 0 = 무제한(옛 동작).
#   시점마다 지속시간이 다르면 (pre 는 파지가 안 되면 영원히 지속) 시점 간 비교가 불가능하다.
_FSTEPS = int(os.environ.get("GS_FORCE_STEPS", "0") or 0)
# ★ 파지 전 외란 전용 계측기: 물체를 cm 단위로 **한 번** 옮긴다.
#   자유 물체에 일정한 힘을 걸면 가속도가 무한정이라(5g x 3.3s = 267 m) 정보가 없다.
#   one-shot 정책의 구조적 질문은 "t=0 계획이 몇 cm 의 물체 위치 오차를 견디는가" 다.
_SHIFT = float(os.environ.get("GS_OBJ_SHIFT", "0") or 0) / 100.0   # cm -> m
_DUMP = os.environ.get("GS_DUMP", "")
_ACTIVE = (_MASS != 1.0) or (_EXTG != 0.0) or (_SHIFT != 0.0) or bool(_DUMP)


def _wrap(Grasp):
    if getattr(Grasp, "_gs_patched", False):
        return
    from isaacgym import gymapi, gymtorch
    import numpy as np
    import torch

    orig_create_envs = Grasp._create_envs
    orig_pre = Grasp.pre_physics_step
    orig_step = Grasp.step
    orig_reward = Grasp.compute_reward

    def _create_envs(self, *a, **kw):
        out = orig_create_envs(self, *a, **kw)
        gym, sim = self.gym, self.sim
        rb_idx, masses = [], []
        for env_ptr in self.envs:
            h = gym.find_actor_handle(env_ptr, "object")
            props = gym.get_actor_rigid_body_properties(env_ptr, h)
            if _MASS != 1.0:
                for p in props:
                    p.mass = p.mass * _MASS
                gym.set_actor_rigid_body_properties(env_ptr, h, props, recomputeInertia=True)
                props = gym.get_actor_rigid_body_properties(env_ptr, h)
            masses.append(sum(p.mass for p in props))
            rb_idx.append(gym.get_actor_rigid_body_index(env_ptr, h, 0, gymapi.DOMAIN_SIM))
        self._gs_obj_rb = torch.tensor(rb_idx, dtype=torch.long, device=self.device)
        self._gs_obj_mass = torch.tensor(masses, dtype=torch.float, device=self.device)
        self._gs_n_rb = gym.get_sim_rigid_body_count(sim)
        self._gs_force = torch.zeros((self._gs_n_rb, 3), dtype=torch.float, device=self.device)
        self._gs_dir = None
        self._gs_applied = 0
        self._gs_shifted = 0
        print("[gs] mass x%.3g (obj mass mean %.4f kg) | ext_g %.3g (%s, phase=%s) | envs %d"
              % (_MASS, self._gs_obj_mass.mean().item(), _EXTG, _MODE,
                 _PHASE + ("" if _FSTEPS == 0 else "/%dstep" % _FSTEPS), self.num_envs), flush=True)
        if _SHIFT != 0.0:
            print("[gs] object shift %.1f cm (계획 확정 직후 1회)" % (_SHIFT * 100), flush=True)
        return out

    def _gs_phase_mask(self):
        """외란을 걸 시점 마스크. ★ 성공 판정식(reward.py:reward_binary)이 쓰는 값을
        그대로 재계산해서 쓴다 — 별도 기준을 만들면 "언제부터 파지인가"가 어긋난다.

          pre   : 손이 아직 물체 근처가 아님        (flag == False)
          grasp : 손은 닿았고 아직 안 들림          (flag & delta_z <= 0.1)
          post  : 들림                              (delta_z > 0.1)
        """
        obj = self.object_pos
        delta_z = obj[:, 2] - self.object_init_states[:, 2]
        palm_d = torch.norm(obj - self.palm_center_pos, dim=-1)
        ft_d = torch.zeros_like(delta_z)
        for i in range(self.fingertip_pos.shape[-2]):
            ft_d += torch.norm(self.fingertip_pos[:, i, :] - obj, dim=-1)
        ft_d = torch.clamp(ft_d, max=3.0)
        flag = (ft_d <= 0.12 * self.num_fingers) | (palm_d <= 0.15)
        lifted = delta_z > 0.1
        if _PHASE == "pre":
            return ~flag
        if _PHASE == "grasp":
            return flag & (~lifted)
        if _PHASE == "all":
            return torch.ones_like(lifted)
        return lifted                      # "post" (기본)

    def _gs_apply_force(self):
        """외력을 물체에 건다. ★ simulate() 호출 직전마다 불려야 한다 — IsaacGym 의
        apply_rigid_body_force_tensors 는 **다음 simulate 1회에만** 적용되기 때문이다.
        grasp.py 의 step() 은 simulate 를 decimation(20)회 부르므로, pre_physics_step 에서
        한 번만 걸면 의도한 힘의 1/20 만 걸린다 (실측으로 확인된 계측기 결함)."""
        if _EXTG == 0.0:
            return
        if _MODE == "iso":
            v = torch.randn(self.num_envs, 3, device=self.device)
            dirs = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            dirs = self._gs_dir
        mask = self._gs_phase_mask()
        if _FSTEPS > 0:
            # 시점에 처음 진입한 env-step 을 기록하고, 그로부터 _FSTEPS 스텝 동안만 발화한다.
            if not hasattr(self, "_gs_t0"):
                self._gs_t0 = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            fresh = mask & (self._gs_t0 < 0)
            self._gs_t0 = torch.where(fresh, self.progress_buf.long(), self._gs_t0)
            self._gs_t0 = torch.where(self.progress_buf == 0,
                                      torch.full_like(self._gs_t0, -1), self._gs_t0)
            within = (self._gs_t0 >= 0) & ((self.progress_buf.long() - self._gs_t0) < _FSTEPS)
            mask = mask & within
        mag = self._gs_obj_mass * 9.81 * _EXTG * mask.float()
        self._gs_force.zero_()
        self._gs_force[self._gs_obj_rb] = dirs * mag.unsqueeze(-1)
        self._gs_applied += int(mask.sum().item())   # 발화한 env-step 누적 (계측기 생존 확인용)
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self._gs_force), None, gymapi.ENV_SPACE)

    class _GymProxy(object):
        """self.gym 을 감싸 simulate() 직전에 외력을 재적용한다. 남의 step() 을 복제하지 않는다."""
        __slots__ = ("_g", "_task")

        def __init__(self, gym, task):
            object.__setattr__(self, "_g", gym)
            object.__setattr__(self, "_task", task)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_g"), name)

        def simulate(self, *a, **kw):
            t = object.__getattribute__(self, "_task")
            t._gs_apply_force()
            return object.__getattribute__(self, "_g").simulate(*a, **kw)

    def pre_physics_step(self, actions):
        out = orig_pre(self, actions)
        if _EXTG != 0.0:
            # 에피소드마다 고정된 수평 방향 (= 옮기는 동작이 만드는 관성력). progress 0 에서 재추첨.
            if self._gs_dir is None or bool((self.progress_buf == 0).any()):
                ang = torch.rand(self.num_envs, device=self.device) * (2 * np.pi)
                d = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
                if self._gs_dir is None:
                    self._gs_dir = d
                else:
                    m = (self.progress_buf == 0).unsqueeze(-1)
                    self._gs_dir = torch.where(m, d, self._gs_dir)
        return out

    def _gs_shift_object(self):
        """계획 확정 직후(progress_buf==1) 물체를 수평 랜덤 방향으로 _SHIFT m 만큼 1회 이동."""
        if _SHIFT == 0.0:
            return
        ids = (self.progress_buf == 1).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        ang = torch.rand(ids.numel(), device=self.device) * (2 * np.pi)
        obj = self.object_indices[ids]
        self.root_state_tensor[obj, 0] += torch.cos(ang) * _SHIFT
        self.root_state_tensor[obj, 1] += torch.sin(ang) * _SHIFT
        self.root_state_tensor[obj, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(obj.to(torch.int32)), obj.numel())
        self._gs_shifted += ids.numel()

    def step(self, actions):
        self._gs_shift_object()
        if _EXTG == 0.0:
            return orig_step(self, actions)
        real = self.gym
        self.gym = _GymProxy(real, self)          # simulate 20회 전부에 힘이 걸린다
        try:
            return orig_step(self, actions)
        finally:
            self.gym = real

    def compute_reward(self):
        out = orig_reward(self)
        if _DUMP:
            judge = int(self.max_episode_length) - 2
            if bool((self.progress_buf == judge).any()):
                rec = getattr(self, "_gs_rec", [])
                rec.append(self.successes.detach().cpu().numpy().copy())
                self._gs_rec = rec
                n_obj = getattr(self, "_gs_n_obj", None) or len(getattr(self, "object_pcls", [])) or 1
                np.savez(_DUMP,
                         successes=np.stack(rec),                       # [round, env]
                         object_id=np.arange(self.num_envs) % n_obj,    # env i -> 물체 i%n (grasp.py:399)
                         n_obj=n_obj, tau=int(self.max_episode_length),
                         mass_scale=_MASS, ext_g=_EXTG, phase=_PHASE, force_steps=_FSTEPS, shift_m=_SHIFT,
                         shifts=getattr(self, "_gs_shifted", 0),
                         obj_mass_kg=float(self._gs_obj_mass.mean().item()),
                         force_applies=getattr(self, "_gs_applied", 0))
                print("[gs] force applies so far: %d" % getattr(self, "_gs_applied", 0), flush=True)
        return out

    Grasp._create_envs = _create_envs
    Grasp._gs_apply_force = _gs_apply_force
    Grasp._gs_phase_mask = _gs_phase_mask
    Grasp._gs_shift_object = _gs_shift_object
    Grasp.pre_physics_step = pre_physics_step
    Grasp.step = step
    Grasp.compute_reward = compute_reward
    Grasp._gs_patched = True
    print("[gs] Grasp patched", flush=True)


def install():
    """`tasks.grasp` 가 로드되는 순간 래핑하는 지연 훅 (시작 시점엔 isaacgym 이 없다)."""
    if not _ACTIVE:
        return
    import importlib.abc, importlib.machinery, sys

    TARGET = "tasks.grasp"

    class _Hook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.machinery.PathFinder.find_spec(name, path)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None:
                return None
            inner = spec.loader

            class _L(importlib.abc.Loader):
                def create_module(s, sp):
                    return inner.create_module(sp)

                def exec_module(s, mod):
                    inner.exec_module(mod)
                    G = getattr(mod, "Grasp", None)
                    if G is not None:
                        _wrap(G)

            spec.loader = _L()
            return spec

    sys.meta_path.insert(0, _Hook())
