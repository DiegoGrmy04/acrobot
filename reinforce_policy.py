"""REINFORCE policy network for the Triple Acrobot (discrete actions).

Notation follows Sutton & Barto (2018), Chapter 13.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Maps observations to a categorical distribution over actions.

    pi_theta(a | s) where theta are the MLP weights.
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
        logits = self.forward(obs)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Discounted Monte-Carlo returns G_t = sum_{k>=t} gamma^(k-t) * r_k.

    Computed in reverse so the cost is O(T) instead of O(T^2).
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)
