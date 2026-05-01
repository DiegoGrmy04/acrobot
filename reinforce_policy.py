"""
REINFORCE policy network for the Triple Acrobot.

REINFORCE (Williams, 1992) is the canonical Monte-Carlo policy gradient method:
no critic, no clipping. We use it to isolate the contribution of the actor-critic
machinery in PPO — same reward shaping, same network width, same env, only the
update rule changes.

Notation (Sutton & Barto, 2018, ch. 13):
    pi_theta(a|s) : stochastic policy parametrized by a neural network.
    G_t = sum_{k=t}^{T} gamma^{k-t} * r_k : Monte-Carlo return from step t.
    Gradient estimator (REINFORCE with baseline b):
        grad J(theta) ~= E[ sum_t (G_t - b) * grad log pi_theta(a_t|s_t) ]
    We use the per-batch return mean as baseline b (cf. S&B §13.4) and
    additionally normalize advantages for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """MLP policy: obs -> logits over discrete actions.

    The hidden width (64) matches stable-baselines3's default MlpPolicy used by
    DQN/PPO in this project, so the comparison isolates the algorithm rather than
    network capacity.
    """

    def __init__(self, obs_dim: int = 9, n_actions: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> int:
        """Single-step action prediction (used by enjoy_agent.py)."""
        logits = self.forward(obs)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Discounted Monte-Carlo returns G_t = sum_{k>=t} gamma^{k-t} r_k.

    Computed in reverse for O(T) instead of O(T^2).
    """
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)
