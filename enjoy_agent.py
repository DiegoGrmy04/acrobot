"""Watch a trained agent play the Triple Acrobot.

Usage:
    python enjoy_agent.py            # interactive menu
    python enjoy_agent.py 5          # run choice #5 directly
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from reinforce_policy import PolicyNetwork
from triple_acrobot import TripleAcrobotEnv

ROOT = Path(__file__).resolve().parent


@dataclass
class ModelChoice:
    label: str
    family: str  # "DQN" | "PPO" | "REINFORCE"
    path: Path


CHOICES = [
    ModelChoice("DQN — sparse",                       "DQN",       ROOT / "results_dqn_sparse/models/best_model.zip"),
    ModelChoice("DQN — shaped",                       "DQN",       ROOT / "results_dqn_shaped/models/best_model.zip"),
    ModelChoice("PPO — sparse",                       "PPO",       ROOT / "results_ppo_sparse/models/best_model.zip"),
    ModelChoice("PPO — shaped",                       "PPO",       ROOT / "results_ppo_shaped/models/best_model.zip"),
    ModelChoice("PPO — PBRS",                         "PPO",       ROOT / "results_ppo_pbrs/models/best_model.zip"),
    ModelChoice("REINFORCE — shaped",                 "REINFORCE", ROOT / "results_reinforce/models/best_modelREINFORCE.pt"),
    ModelChoice("[Legacy] DQN du collègue",           "DQN",       ROOT / "models/best_modelDQN.zip"),
    ModelChoice("[Legacy] PPO du collègue",           "PPO",       ROOT / "models/best_modelPPO.zip"),
]


def make_env():
    env = TripleAcrobotEnv(render_mode="human")
    env = TimeLimit(env, max_episode_steps=1000)
    return env


def load_policy(choice: ModelChoice):
    """Load a trained model. Returns a function obs_batch -> action_batch."""
    if not choice.path.exists():
        raise FileNotFoundError(f"Model not found: {choice.path}")

    if choice.family == "DQN":
        model = DQN.load(str(choice.path))

        def predict(obs_batch):
            action, _ = model.predict(obs_batch, deterministic=True)
            return action

        return predict

    if choice.family == "PPO":
        model = PPO.load(str(choice.path))

        def predict(obs_batch):
            action, _ = model.predict(obs_batch, deterministic=True)
            return action

        return predict

    # REINFORCE: a custom PyTorch network.
    ckpt = torch.load(str(choice.path), map_location="cpu", weights_only=False)
    hidden = ckpt.get("hyper", {}).get("hidden", 64)
    policy = PolicyNetwork(obs_dim=9, n_actions=3, hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    def predict(obs_batch):
        obs_t = torch.as_tensor(obs_batch[0], dtype=torch.float32)
        return [policy.predict(obs_t, deterministic=True)]

    return predict


def show_menu() -> int:
    print("\n=== Triple Acrobot — choose a model ===")
    for i, choice in enumerate(CHOICES, start=1):
        marker = "✓" if choice.path.exists() else "✗"
        print(f"  {marker} {i}. {choice.label}")
    while True:
        s = input(f"Your choice [1-{len(CHOICES)}]: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(CHOICES):
            return int(s) - 1
        print("Invalid choice.")


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        idx = int(sys.argv[1]) - 1
    else:
        idx = show_menu()

    if not (0 <= idx < len(CHOICES)):
        print(f"Invalid index. Must be between 1 and {len(CHOICES)}.")
        return 1

    choice = CHOICES[idx]
    print(f"\nLoading: {choice.label}")
    print(f"  ({choice.path.relative_to(ROOT)})")
    try:
        predict = load_policy(choice)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    env = DummyVecEnv([make_env])
    obs = env.reset()
    print("\nStarting animation (Ctrl+C to stop)...\n")

    episode_steps, episode_idx = 0, 1
    try:
        for _ in range(5000):
            action = predict(obs)
            obs, _, done, _ = env.step(action)
            episode_steps += 1
            time.sleep(0.02)
            if done[0]:
                outcome = "success" if episode_steps < 1000 else "timeout"
                print(f"  ep.{episode_idx:2d}  steps={episode_steps:4d}  {outcome}")
                episode_idx += 1
                episode_steps = 0
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
