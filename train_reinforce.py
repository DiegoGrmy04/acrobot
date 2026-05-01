"""Train REINFORCE locally on the Triple Acrobot (sanity / short runs).

For full runs (1-2M steps) prefer kaggle/reinforce_kaggle.py on a GPU.

Usage:
    python train_reinforce.py
"""
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from numpy import cos
from torch.utils.tensorboard import SummaryWriter

from reinforce_policy import PolicyNetwork, compute_returns
from triple_acrobot import TripleAcrobotEnv


HYPER = dict(
    seed=0,
    total_episodes=4000,
    batch_episodes=8,
    gamma=0.99,
    learning_rate=3e-4,
    hidden=64,
    max_episode_steps=1000,
    entropy_coef=0.01,
    grad_clip=0.5,
    log_dir="./tensorboard_logs/REINFORCE_TripleAcrobot",
    save_path="./models/best_modelREINFORCE.pt",
)


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


def make_env(seed: int) -> gym.Env:
    env = TripleAcrobotEnv(render_mode=None)
    env = TripleAcrobotRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=HYPER["max_episode_steps"])
    env.reset(seed=seed)
    return env


def run_episode(env: gym.Env, policy: PolicyNetwork, device: torch.device):
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
    return log_probs, entropies, rewards, success


def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs(HYPER["log_dir"], exist_ok=True)

    torch.manual_seed(HYPER["seed"])
    np.random.seed(HYPER["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = make_env(HYPER["seed"])
    policy = PolicyNetwork(obs_dim=9, n_actions=3, hidden=HYPER["hidden"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=HYPER["learning_rate"])
    writer = SummaryWriter(HYPER["log_dir"])

    best_ep_len = float("inf")
    total_env_steps = 0
    n_updates = HYPER["total_episodes"] // HYPER["batch_episodes"]
    t0 = time.time()

    for update in range(n_updates):
        batch_log_probs, batch_returns, batch_entropies = [], [], []
        ep_lengths, ep_successes, ep_returns = [], [], []

        for _ in range(HYPER["batch_episodes"]):
            log_probs, entropies, rewards, success = run_episode(env, policy, device)
            returns = compute_returns(rewards, HYPER["gamma"]).to(device)
            batch_log_probs.append(torch.stack(log_probs))
            batch_returns.append(returns)
            batch_entropies.append(torch.stack(entropies))
            ep_lengths.append(len(rewards))
            ep_successes.append(success)
            ep_returns.append(float(sum(rewards)))
            total_env_steps += len(rewards)

        log_probs_t = torch.cat(batch_log_probs)
        returns_t = torch.cat(batch_returns)
        entropies_t = torch.cat(batch_entropies)

        # REINFORCE with batch baseline (Sutton & Barto, eq. 13.11) + advantage normalization.
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
        writer.add_scalar("rollout/ep_len_mean", ep_len_mean, total_env_steps)
        writer.add_scalar("rollout/ep_rew_mean", ep_return_mean, total_env_steps)
        writer.add_scalar("rollout/success_rate", success_rate, total_env_steps)
        writer.add_scalar("train/policy_loss", float(policy_loss.item()), total_env_steps)
        writer.add_scalar("train/entropy", float(entropy_bonus.item()), total_env_steps)
        writer.add_scalar("train/grad_norm", float(grad_norm), total_env_steps)

        if update % 5 == 0:
            elapsed = time.time() - t0
            fps = int(total_env_steps / elapsed) if elapsed > 0 else 0
            print(
                f"update={update:4d} steps={total_env_steps:7d} "
                f"ep_len={ep_len_mean:6.1f} return={ep_return_mean:8.2f} "
                f"success={success_rate:.2f} grad={grad_norm:5.2f} fps={fps}"
            )

        if success_rate > 0 and ep_len_mean < best_ep_len:
            best_ep_len = ep_len_mean
            torch.save({"state_dict": policy.state_dict(), "hyper": HYPER}, HYPER["save_path"])

    writer.close()
    env.close()
    print(f"Done. Best ep_len_mean = {best_ep_len:.1f}")


if __name__ == "__main__":
    train()
