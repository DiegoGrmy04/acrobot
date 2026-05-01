"""
Local REINFORCE training for the Triple Acrobot (sanity check / short runs).

For full training (1-2M env steps), prefer running kaggle/reinforce_kaggle.py
on a Kaggle GPU notebook — same script, no external imports required.

Usage:
    python train_reinforce.py
"""

from __future__ import annotations

import os
import time
from numpy import cos

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from torch.utils.tensorboard import SummaryWriter

from reinforce_policy import PolicyNetwork, compute_returns
from triple_acrobot import TripleAcrobotEnv


# ----- Hyperparamètres figés en haut du script (cf. CLAUDE.md §7) -----
HYPER = dict(
    seed=0,
    total_episodes=4000,
    batch_episodes=8,            # nb d'épisodes accumulés avant chaque update
    gamma=0.99,
    learning_rate=3e-4,
    hidden=64,
    max_episode_steps=1000,
    entropy_coef=0.01,           # encourage l'exploration tant que la politique est plate
    grad_clip=0.5,
    log_dir="./tensorboard_logs/REINFORCE_TripleAcrobot",
    save_path="./models/best_modelREINFORCE.pt",
)


class TripleAcrobotRewardWrapper(gym.Wrapper):
    """Même reward shaping que celui utilisé pour PPO, pour une comparaison équitable."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        s = self.unwrapped.state
        hauteur = -cos(s[0]) - cos(s[1] + s[0]) - cos(s[2] + s[1] + s[0])
        shaped = -1.0 + hauteur
        if terminated:
            shaped += 100.0
        return obs, shaped, terminated, truncated, info


def make_env(seed: int) -> gym.Env:
    env = TripleAcrobotEnv(render_mode=None)
    env = TripleAcrobotRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=HYPER["max_episode_steps"])
    env.reset(seed=seed)
    return env


def run_episode(env: gym.Env, policy: PolicyNetwork, device: torch.device):
    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[float] = []
    raw_length = 0
    success = False
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        dist = policy.distribution(obs_t)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        obs, r, term, trunc, _ = env.step(int(action.item()))
        rewards.append(float(r))
        raw_length += 1
        if term:
            success = True
        done = term or trunc
    return log_probs, entropies, rewards, raw_length, success


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

    best_eval_len = float("inf")
    total_env_steps = 0
    t0 = time.time()

    for update in range(HYPER["total_episodes"] // HYPER["batch_episodes"]):
        batch_log_probs: list[torch.Tensor] = []
        batch_returns: list[torch.Tensor] = []
        batch_entropies: list[torch.Tensor] = []
        ep_lengths: list[int] = []
        ep_successes: list[bool] = []
        ep_returns_undisc: list[float] = []

        # 1) Collecte d'un batch d'épisodes (Monte-Carlo, on-policy)
        for _ in range(HYPER["batch_episodes"]):
            log_probs, entropies, rewards, length, success = run_episode(env, policy, device)
            returns = compute_returns(rewards, HYPER["gamma"]).to(device)
            batch_log_probs.append(torch.stack(log_probs))
            batch_returns.append(returns)
            batch_entropies.append(torch.stack(entropies))
            ep_lengths.append(length)
            ep_successes.append(success)
            ep_returns_undisc.append(float(sum(rewards)))
            total_env_steps += length

        log_probs_cat = torch.cat(batch_log_probs)
        returns_cat = torch.cat(batch_returns)
        entropy_cat = torch.cat(batch_entropies)

        # 2) Baseline + normalisation des avantages (réduction de variance)
        advantages = (returns_cat - returns_cat.mean()) / (returns_cat.std() + 1e-8)

        # 3) Loss REINFORCE = -E[ A_t * log pi(a_t|s_t) ] + bonus d'entropie
        policy_loss = -(log_probs_cat * advantages.detach()).mean()
        entropy_loss = -entropy_cat.mean()
        loss = policy_loss + HYPER["entropy_coef"] * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), HYPER["grad_clip"])
        optimizer.step()

        # 4) Logging
        ep_len_mean = float(np.mean(ep_lengths))
        ep_ret_mean = float(np.mean(ep_returns_undisc))
        success_rate = float(np.mean(ep_successes))
        writer.add_scalar("rollout/ep_len_mean", ep_len_mean, total_env_steps)
        writer.add_scalar("rollout/ep_rew_mean", ep_ret_mean, total_env_steps)
        writer.add_scalar("rollout/success_rate", success_rate, total_env_steps)
        writer.add_scalar("train/policy_loss", float(policy_loss.item()), total_env_steps)
        writer.add_scalar("train/entropy", float(-entropy_loss.item()), total_env_steps)
        writer.add_scalar("train/grad_norm", float(grad_norm), total_env_steps)

        if update % 5 == 0:
            elapsed = time.time() - t0
            fps = int(total_env_steps / elapsed) if elapsed > 0 else 0
            print(
                f"update={update:4d} steps={total_env_steps:7d} "
                f"ep_len={ep_len_mean:6.1f} ret={ep_ret_mean:8.2f} "
                f"succ={success_rate:.2f} grad={grad_norm:5.2f} fps={fps}"
            )

        # Sauvegarde du meilleur modèle (par durée d'épisode moyenne)
        if ep_len_mean < best_eval_len and success_rate > 0:
            best_eval_len = ep_len_mean
            torch.save(
                {"state_dict": policy.state_dict(), "hyper": HYPER},
                HYPER["save_path"],
            )

    writer.close()
    env.close()
    print(f"Entraînement terminé. Meilleur ep_len_mean = {best_eval_len:.1f}")


if __name__ == "__main__":
    train()
