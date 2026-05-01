"""Compare the 6 trained models on the Triple Acrobot.

Outputs (in analysis/):
  - results_table.csv / .tex   : deterministic eval on N episodes
  - learning_curves.pdf        : training curves (TensorBoard events)
  - eval_summary.pdf           : final ep_len ± std bar chart

Usage:
    python analysis/compare_models.py              # full eval (100 episodes)
    python analysis/compare_models.py --n-eval 10  # quick smoke test
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reinforce_policy import PolicyNetwork  # noqa: E402
from triple_acrobot import TripleAcrobotEnv  # noqa: E402


@dataclass
class Run:
    name: str
    label: str
    algo: str
    reward: str
    model_path: Path
    tb_dir: Path
    color: str


RUNS = [
    Run("dqn_sparse", "DQN — sparse", "DQN", "sparse",
        ROOT / "results_dqn_sparse/models/best_model.zip",
        ROOT / "results_dqn_sparse/tensorboard_logs/DQN_Sparse_TripleAcrobot_1",
        "#d62728"),
    Run("dqn_shaped", "DQN — shaped", "DQN", "shaped",
        ROOT / "results_dqn_shaped/models/best_model.zip",
        ROOT / "results_dqn_shaped/tensorboard_logs/DQN_Shaped_TripleAcrobot_1",
        "#ff7f0e"),
    Run("ppo_sparse", "PPO — sparse", "PPO", "sparse",
        ROOT / "results_ppo_sparse/models/best_model.zip",
        ROOT / "results_ppo_sparse/tensorboard_logs/PPO_Sparse_TripleAcrobot_1",
        "#1f77b4"),
    Run("ppo_shaped", "PPO — shaped", "PPO", "shaped",
        ROOT / "results_ppo_shaped/models/best_model.zip",
        ROOT / "results_ppo_shaped/tensorboard_logs/PPO_Shaped_TripleAcrobot_1",
        "#2ca02c"),
    Run("ppo_pbrs", "PPO — PBRS", "PPO", "PBRS",
        ROOT / "results_ppo_pbrs/models/best_model.zip",
        ROOT / "results_ppo_pbrs/tensorboard_logs/PPO_PBRS_TripleAcrobot_1",
        "#9467bd"),
    Run("reinforce_shaped", "REINFORCE — shaped", "REINFORCE", "shaped",
        ROOT / "results_reinforce/models/best_modelREINFORCE.pt",
        ROOT / "results_reinforce/tensorboard_logs/REINFORCE_TripleAcrobot",
        "#8c564b"),
]


def make_eval_env() -> TimeLimit:
    """Eval env: native sparse reward (-1 / 0). Shaping is only a training trick."""
    env = TripleAcrobotEnv(render_mode=None)
    env = TimeLimit(env, max_episode_steps=1000)
    return env


def load_predict_fn(run: Run):
    """Return a function obs -> action for any algorithm family."""
    if run.algo == "DQN":
        model = DQN.load(str(run.model_path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if run.algo == "PPO":
        model = PPO.load(str(run.model_path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if run.algo == "REINFORCE":
        ckpt = torch.load(str(run.model_path), map_location="cpu", weights_only=False)
        hidden = ckpt.get("hyper", {}).get("hidden", 64)
        policy = PolicyNetwork(obs_dim=9, n_actions=3, hidden=hidden)
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        return lambda obs: policy.predict(
            torch.as_tensor(np.asarray(obs), dtype=torch.float32),
            deterministic=True,
        )
    raise ValueError(run.algo)


def evaluate(run: Run, n_episodes: int, seed: int = 12345) -> dict:
    env = make_eval_env()
    predict = load_predict_fn(run)
    ep_lengths, ep_successes = [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done, ep_length, success = False, 0, False
        while not done:
            action = predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            ep_length += 1
            if terminated:
                success = True
            done = terminated or truncated
        ep_lengths.append(ep_length)
        ep_successes.append(success)
    env.close()
    return {
        "ep_len_mean": float(np.mean(ep_lengths)),
        "ep_len_std": float(np.std(ep_lengths)),
        "success_rate": float(np.mean(ep_successes)),
        "n_episodes": n_episodes,
    }


def read_tb_scalar(tb_dir: Path, tag: str):
    """Return (steps, values) for a TensorBoard scalar tag, or None if missing."""
    if not tb_dir.exists():
        return None
    candidates = [tb_dir] + [p for p in tb_dir.iterdir() if p.is_dir()]
    for d in candidates:
        try:
            ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        if tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            return steps, values
    return None


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Edge-aware moving average (no zero-padding artifact at the borders)."""
    if len(y) < 2 or window <= 1:
        return y
    w = min(window, len(y))
    kernel = np.ones(w)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y, dtype=float), kernel, mode="same")
    return num / den


def write_table(results: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "results_table.csv"
    tex_path = out_dir / "results_table.tex"
    fields = ["run", "algo", "reward", "n_episodes",
              "ep_len_mean", "ep_len_std", "success_rate"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"[OK] {csv_path}")

    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
        f.write("Algorithm & Reward & $\\bar{\\ell}$ (steps) & $\\sigma_\\ell$ & Success rate \\\\\n")
        f.write("\\midrule\n")
        for r in results:
            f.write(
                f"{r['algo']} & {r['reward']} & "
                f"{r['ep_len_mean']:.1f} & {r['ep_len_std']:.1f} & "
                f"{r['success_rate']*100:.0f}\\% \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[OK] {tex_path}")


def save_figure(fig, out_dir: Path, name: str) -> None:
    """Save as PDF (vector, for LaTeX) and PNG (preview)."""
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"[OK] {pdf_path}")
    print(f"[OK] {png_path}")


def plot_learning_curves(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for run in RUNS:
        data = read_tb_scalar(run.tb_dir, "rollout/ep_len_mean")
        if data is None:
            print(f"[skip] {run.name}: no ep_len_mean tag")
            continue
        steps, values = data
        ax1.plot(steps, smooth(values, window=5), label=run.label, color=run.color, lw=1.6)
    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Mean episode length")
    ax1.set_title("Episode length during training")
    ax1.set_ylim(0, 1050)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    for run in RUNS:
        data = read_tb_scalar(run.tb_dir, "rollout/ep_rew_mean")
        if data is None:
            continue
        steps, values = data
        ax2.plot(steps, smooth(values, window=5), label=run.label, color=run.color, lw=1.6)
    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel("Mean episode return (training reward)")
    ax2.set_title("Training reward (shaping-dependent scale)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    save_figure(fig, out_dir, "learning_curves")
    plt.close(fig)


def plot_eval_summary(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = [r["label"] for r in results]
    means = [r["ep_len_mean"] for r in results]
    stds = [r["ep_len_std"] for r in results]
    colors = [r["color"] for r in results]
    x = np.arange(len(results))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", lw=0.6, capsize=4)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean episode length")
    ax.set_title(
        f"Deterministic evaluation on sparse reward "
        f"(n = {results[0]['n_episodes']} episodes per run)"
    )
    ax.set_ylim(0, 1150)
    ax.axhline(1000, ls="--", color="grey", lw=0.7, alpha=0.7)
    ax.text(len(results) - 0.5, 1015, "TimeLimit", fontsize=8, color="grey")
    ax.grid(True, axis="y", alpha=0.3)
    for i, r in enumerate(results):
        ax.text(i, means[i] + stds[i] + 25,
                f"{r['success_rate']*100:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    save_figure(fig, out_dir, "eval_summary")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip eval and only regenerate the learning curves plot.")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent

    print(f"=== Deterministic evaluation on {args.n_eval} episodes per run ===")
    results = []
    for run in RUNS:
        if not run.model_path.exists():
            print(f"[SKIP] {run.name}: model not found ({run.model_path})")
            continue
        if args.skip_eval:
            stats = {"ep_len_mean": np.nan, "ep_len_std": 0.0,
                     "success_rate": np.nan, "n_episodes": 0}
        else:
            print(f"  -> {run.name} ...", flush=True)
            stats = evaluate(run, args.n_eval)
        results.append({
            "run": run.name, "algo": run.algo, "reward": run.reward,
            "label": run.label, "color": run.color, **stats,
        })
        print(f"     ep_len = {stats['ep_len_mean']:.1f} ± {stats['ep_len_std']:.1f}, "
              f"success = {stats['success_rate']*100:.0f}%")

    print("\n=== Generating outputs ===")
    if not args.skip_eval:
        write_table(results, out_dir)
        plot_eval_summary(results, out_dir)
    plot_learning_curves(out_dir)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
