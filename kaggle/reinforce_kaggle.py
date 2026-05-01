"""REINFORCE on the Triple Acrobot — Kaggle GPU notebook.

Self-contained: paste in a Kaggle notebook, set Accelerator = GPU T4 x2,
choose Internet = ON and Persistence = Files only, then Save & Run All.

Notation follows Sutton & Barto (2018), Chapter 13.
"""
# Kaggle dependencies
!pip install -q gymnasium pygame tensorboard

import os
import time
import zipfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env, spaces
from gymnasium.wrappers import TimeLimit
from numpy import cos, pi, sin
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


# Triple Acrobot environment (3-link version of Gym's Acrobot-v1).
# Inlined so the Kaggle notebook is fully self-contained.
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
    """Same dense reward as the PPO run, for a fair comparison."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        s = self.unwrapped.state
        height = -cos(s[0]) - cos(s[1] + s[0]) - cos(s[2] + s[1] + s[0])
        shaped_reward = -1.0 + height
        if terminated:
            shaped_reward += 100.0
        return obs, shaped_reward, terminated, truncated, info


class PolicyNetwork(nn.Module):
    """pi_theta(a | s) — categorical over 3 discrete actions."""

    def __init__(self, obs_dim=9, n_actions=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)

    def distribution(self, obs):
        return Categorical(logits=self.forward(obs))


def compute_returns(rewards, gamma):
    """Discounted Monte-Carlo returns G_t = sum_{k>=t} gamma^(k-t) * r_k."""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


HYPER = dict(
    seed=0,
    total_env_steps=2_000_000,
    batch_episodes=8,
    gamma=0.99,
    learning_rate=3e-4,
    hidden=64,
    max_episode_steps=1000,
    entropy_coef=0.01,
    grad_clip=0.5,
)

RUN_NAME = "reinforce_shaped"
WORK = "/kaggle/working"
os.makedirs(f"{WORK}/models", exist_ok=True)
os.makedirs(f"{WORK}/logs", exist_ok=True)
os.makedirs(f"{WORK}/tensorboard_logs", exist_ok=True)

torch.manual_seed(HYPER["seed"])
np.random.seed(HYPER["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

env = TimeLimit(
    TripleAcrobotRewardWrapper(TripleAcrobotEnv()),
    max_episode_steps=HYPER["max_episode_steps"],
)
env.reset(seed=HYPER["seed"])

policy = PolicyNetwork(9, 3, HYPER["hidden"]).to(device)
optimizer = optim.Adam(policy.parameters(), lr=HYPER["learning_rate"])
writer = SummaryWriter(f"{WORK}/tensorboard_logs/REINFORCE_TripleAcrobot")

total_steps = 0
update = 0
best_ep_len = float("inf")
t0 = time.time()

print("Starting REINFORCE training...")
while total_steps < HYPER["total_env_steps"]:
    batch_log_probs, batch_returns, batch_entropies = [], [], []
    ep_lengths, ep_successes, ep_returns = [], [], []

    for _ in range(HYPER["batch_episodes"]):
        obs, _ = env.reset()
        log_probs, entropies, rewards = [], [], []
        success, done = False, False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist = policy.distribution(obs_t)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(float(reward))
            if terminated:
                success = True
            done = terminated or truncated

        batch_log_probs.append(torch.stack(log_probs))
        batch_returns.append(compute_returns(rewards, HYPER["gamma"]).to(device))
        batch_entropies.append(torch.stack(entropies))
        ep_lengths.append(len(rewards))
        ep_successes.append(success)
        ep_returns.append(sum(rewards))
        total_steps += len(rewards)

    log_probs_t = torch.cat(batch_log_probs)
    returns_t = torch.cat(batch_returns)
    entropies_t = torch.cat(batch_entropies)

    # Batch baseline + advantage normalization (Sutton & Barto §13.4).
    advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    policy_loss = -(log_probs_t * advantages.detach()).mean()
    entropy_bonus = entropies_t.mean()
    loss = policy_loss - HYPER["entropy_coef"] * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), HYPER["grad_clip"])
    optimizer.step()

    ep_len_mean = float(np.mean(ep_lengths))
    ep_return_mean = float(np.mean(ep_returns))
    success_rate = float(np.mean(ep_successes))
    writer.add_scalar("rollout/ep_len_mean", ep_len_mean, total_steps)
    writer.add_scalar("rollout/ep_rew_mean", ep_return_mean, total_steps)
    writer.add_scalar("rollout/success_rate", success_rate, total_steps)
    writer.add_scalar("train/policy_loss", float(policy_loss.item()), total_steps)
    writer.add_scalar("train/entropy", float(entropy_bonus.item()), total_steps)
    writer.add_scalar("train/grad_norm", float(grad_norm), total_steps)

    if update % 5 == 0:
        fps = int(total_steps / max(time.time() - t0, 1e-6))
        print(
            f"update={update:4d} steps={total_steps:7d} ep_len={ep_len_mean:6.1f} "
            f"return={ep_return_mean:8.2f} success={success_rate:.2f} grad={grad_norm:5.2f} fps={fps}"
        )

    if success_rate > 0 and ep_len_mean < best_ep_len:
        best_ep_len = ep_len_mean
        torch.save(
            {"state_dict": policy.state_dict(), "hyper": HYPER},
            f"{WORK}/models/best_modelREINFORCE.pt",
        )

    update += 1

torch.save(
    {"state_dict": policy.state_dict(), "hyper": HYPER},
    f"{WORK}/models/reinforce_triple_acrobot_final.pt",
)
writer.close()
print(f"Done. Best ep_len_mean = {best_ep_len:.1f}")


# Package everything into a single zip for download.
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
