"""DQN on the Triple Acrobot, naive reward shaping — Kaggle GPU notebook.

Self-contained: paste in a Kaggle notebook, set Accelerator = GPU T4 x2,
choose Internet = ON and Persistence = Files only, then Save & Run All.

Reward shaping: R'(s, a, s') = -1 + height(s') + 100 if terminal.
This is NOT potential-based (cf. Ng et al. 1999), so policy invariance is not
guaranteed. See ppo_pbrs_kaggle.py for the conforming variant.
"""
# Kaggle dependencies
!pip install -q gymnasium pygame
!pip install -q "stable-baselines3[extra]" tensorboard

import os
import zipfile

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from gymnasium.wrappers import TimeLimit
from numpy import cos, pi, sin
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


class TripleAcrobotEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    dt = 0.05
    LINK_LENGTH_1, LINK_LENGTH_2, LINK_LENGTH_3 = 1.0, 1.0, 1.0
    LINK_MASS_1, LINK_MASS_2, LINK_MASS_3 = 1.0, 1.0, 1.0
    LINK_COM_POS_1, LINK_COM_POS_2, LINK_COM_POS_3 = 0.5, 0.5, 0.5
    LINK_MOI = 1.0
    MAX_VEL_1, MAX_VEL_2, MAX_VEL_3 = 4 * pi, 9 * pi, 15 * pi
    AVAIL_TORQUE = [-2.0, 0.0, +2.0]

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2, self.MAX_VEL_3],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(6,)).astype(np.float32)
        return self._get_ob(), {}

    def step(self, a):
        s = self.state
        torque = self.AVAIL_TORQUE[a]
        s_augmented = np.append(s, torque)
        ns = self.rk4(self._dsdt, s_augmented, [0, self.dt])

        def wrap(x, m, M):
            diff = M - m
            while x > M:
                x -= diff
            while x < m:
                x += diff
            return x

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = wrap(ns[2], -pi, pi)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[4] = np.clip(ns[4], -self.MAX_VEL_2, self.MAX_VEL_2)
        ns[5] = np.clip(ns[5], -self.MAX_VEL_3, self.MAX_VEL_3)

        self.state = ns
        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0
        return self._get_ob(), reward, terminated, False, {}

    def _get_ob(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), cos(s[2]), sin(s[2]), s[3], s[4], s[5]],
            dtype=np.float32,
        )

    def _terminal(self):
        s = self.state
        height = -cos(s[0]) - cos(s[1] + s[0]) - cos(s[2] + s[1] + s[0])
        return bool(height > 2.0)

    def _dsdt(self, s_augmented):
        g = 9.81
        m1, m2, m3 = self.LINK_MASS_1, self.LINK_MASS_2, self.LINK_MASS_3
        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        lc1, lc2, lc3 = self.LINK_COM_POS_1, self.LINK_COM_POS_2, self.LINK_COM_POS_3
        I1, I2, I3 = self.LINK_MOI, self.LINK_MOI, self.LINK_MOI
        a = s_augmented[-1]
        s = s_augmented[:-1]
        t1, t2, t3 = s[0], s[1], s[2]
        dt1, dt2, dt3 = s[3], s[4], s[5]
        a1, a2, a3 = t1, t1 + t2, t1 + t2 + t3
        da1, da2, da3 = dt1, dt1 + dt2, dt1 + dt2 + dt3

        M11 = m1 * lc1**2 + m2 * l1**2 + m3 * l1**2 + I1
        M22 = m2 * lc2**2 + m3 * l2**2 + I2
        M33 = m3 * lc3**2 + I3
        c12 = m2 * l1 * lc2 + m3 * l1 * l2
        c13 = m3 * l1 * lc3
        c23 = m3 * l2 * lc3
        M12 = c12 * cos(a1 - a2)
        M13 = c13 * cos(a1 - a3)
        M23 = c23 * cos(a2 - a3)

        M = np.array([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])
        V1 = c12 * sin(a1 - a2) * da2**2 + c13 * sin(a1 - a3) * da3**2
        V2 = -c12 * sin(a1 - a2) * da1**2 + c23 * sin(a2 - a3) * da3**2
        V3 = -c13 * sin(a1 - a3) * da1**2 - c23 * sin(a2 - a3) * da2**2
        V = np.array([V1, V2, V3])

        G1 = (m1 * lc1 + m2 * l1 + m3 * l1) * g * sin(a1)
        G2 = (m2 * lc2 + m3 * l2) * g * sin(a2)
        G3 = (m3 * lc3) * g * sin(a3)
        G = np.array([G1, G2, G3])

        Tau = np.array([-a, a, 0.0])
        dd_a = np.linalg.solve(M, Tau - V - G)
        return dt1, dt2, dt3, dd_a[0], dd_a[1] - dd_a[0], dd_a[2] - dd_a[1], 0.0

    def rk4(self, derivs, y0, t):
        yout = np.zeros((len(t), len(y0)), np.float64)
        yout[0] = y0
        for i in np.arange(len(t) - 1):
            dt = t[i + 1] - t[i]
            dt2 = dt / 2.0
            y0 = yout[i]
            k1 = np.asarray(derivs(y0))
            k2 = np.asarray(derivs(y0 + dt2 * k1))
            k3 = np.asarray(derivs(y0 + dt2 * k2))
            k4 = np.asarray(derivs(y0 + dt * k3))
            yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout[-1][:6]


class TripleAcrobotRewardWrapper(gym.Wrapper):
    """Naive dense reward: R' = -1 + height(s') + 100 on success."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        s = self.unwrapped.state
        height = -cos(s[0]) - cos(s[1] + s[0]) - cos(s[2] + s[1] + s[0])
        shaped_reward = -1.0 + height
        if terminated:
            shaped_reward += 100.0
        return obs, shaped_reward, terminated, truncated, info


RUN_NAME = "dqn_shaped"
WORK = "/kaggle/working"
os.makedirs(f"{WORK}/models", exist_ok=True)
os.makedirs(f"{WORK}/logs", exist_ok=True)
os.makedirs(f"{WORK}/tensorboard_logs", exist_ok=True)


def make_env():
    env = TripleAcrobotEnv(render_mode=None)
    env = TripleAcrobotRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=1000)
    return env


vec_env = make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv)
eval_env = make_env()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{WORK}/models/",
    log_path=f"{WORK}/logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

model = DQN(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=128,
    exploration_fraction=0.5,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log=f"{WORK}/tensorboard_logs/",
    device="cuda",
)

print("Starting DQN training (shaped reward)...")
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
    tb_log_name="DQN_Shaped_TripleAcrobot",
)
print("Done.")
model.save(f"{WORK}/models/dqn_shaped_triple_acrobot_final")


zip_path = f"{WORK}/{RUN_NAME}_artifacts.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ["models", "logs", "tensorboard_logs"]:
        sub_dir = f"{WORK}/{sub}"
        if not os.path.isdir(sub_dir):
            continue
        for root, _, files in os.walk(sub_dir):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, arcname=os.path.relpath(full, WORK))

print("\nContents of /kaggle/working/:")
for root, _, files in os.walk(WORK):
    for f in files:
        path = os.path.join(root, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {size_kb:8.1f} KB  {path}")
print(f"\nDownload: {zip_path}")
