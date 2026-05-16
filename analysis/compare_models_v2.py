"""Compare the trained models on the Triple Acrobot — version 2 (multi-seed + bignet).

Extends analysis/compare_models.py with 4 additional runs:
  - PPO+PBRS seed=1 and seed=2 (multi-seed validation of the main config)
  - DQN sparse seed=1 (multi-seed validation of the deadly-triad narrative)
  - PPO+PBRS bignet [256, 256] (NN architecture ablation)

Outputs (in analysis/, suffix _v2 to keep originals intact):
  - results_table_v2.csv / .tex   : per-run deterministic eval (10 rows)
  - results_aggregate_v2.csv      : aggregated mean±std for multi-seed configs
  - learning_curves_v2.pdf        : training curves (10 runs, multi-seed marked)
  - eval_summary_v2.pdf           : final ep_len ± std bar chart (10 runs)
  - multiseed_curves_v2.pdf       : PPO+PBRS and DQN sparse as mean±std bands

Usage:
    python analysis/compare_models_v2.py              # full eval (100 episodes)
    python analysis/compare_models_v2.py --n-eval 10  # quick smoke test
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
    group: str = ""  # used to aggregate multi-seed runs


# 6 baseline runs (originaux, déjà dans results_table.csv)
RUNS_BASE = [
    Run("dqn_sparse", "DQN — sparse (seed 0)", "DQN", "sparse",
        ROOT / "results_dqn_sparse/models/best_model.zip",
        ROOT / "results_dqn_sparse/tensorboard_logs/DQN_Sparse_TripleAcrobot_1",
        "#d62728", group="dqn_sparse"),
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
    Run("ppo_pbrs", "PPO — PBRS (seed 0)", "PPO", "PBRS",
        ROOT / "results_ppo_pbrs/models/best_model.zip",
        ROOT / "results_ppo_pbrs/tensorboard_logs/PPO_PBRS_TripleAcrobot_1",
        "#9467bd", group="ppo_pbrs"),
    Run("reinforce_shaped", "REINFORCE — shaped", "REINFORCE", "shaped",
        ROOT / "results_reinforce/models/best_modelREINFORCE.pt",
        ROOT / "results_reinforce/tensorboard_logs/REINFORCE_TripleAcrobot",
        "#8c564b"),
]

# 4 nouveaux runs lancés le 2026-05-16
RUNS_NEW = [
    Run("ppo_pbrs_seed1", "PPO — PBRS (seed 1)", "PPO", "PBRS",
        ROOT / "results_ppo_pbrs_seed1/models/best_model.zip",
        ROOT / "results_ppo_pbrs_seed1/tensorboard_logs/PPO_PBRS_TripleAcrobot_1",
        "#9467bd", group="ppo_pbrs"),
    Run("ppo_pbrs_seed2", "PPO — PBRS (seed 2)", "PPO", "PBRS",
        ROOT / "results_ppo_pbrs_seed2/models/best_model.zip",
        ROOT / "results_ppo_pbrs_seed2/tensorboard_logs/PPO_PBRS_TripleAcrobot_1",
        "#9467bd", group="ppo_pbrs"),
    Run("dqn_sparse_seed1", "DQN — sparse (seed 1)", "DQN", "sparse",
        ROOT / "results_dqn_sparse_seed1/models/best_model.zip",
        ROOT / "results_dqn_sparse_seed1/tensorboard_logs/DQN_Sparse_TripleAcrobot_1",
        "#d62728", group="dqn_sparse"),
    Run("ppo_pbrs_bignet", "PPO — PBRS bignet [256,256]", "PPO", "PBRS",
        ROOT / "results_ppo_pbrs_bignet/models/best_model.zip",
        ROOT / "results_ppo_pbrs_bignet/tensorboard_logs/PPO_PBRS_BigNet_TripleAcrobot_1",
        "#e377c2"),
]

RUNS = RUNS_BASE + RUNS_NEW


def make_eval_env() -> TimeLimit:
    env = TripleAcrobotEnv(render_mode=None)
    env = TimeLimit(env, max_episode_steps=1000)
    return env


def load_predict_fn(run: Run):
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
    if len(y) < 2 or window <= 1:
        return y
    w = min(window, len(y))
    kernel = np.ones(w)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y, dtype=float), kernel, mode="same")
    return num / den


def write_table(results: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "results_table_v2.csv"
    tex_path = out_dir / "results_table_v2.tex"
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


def write_aggregate(results: list[dict], out_dir: Path) -> None:
    """Aggregate per-seed runs into mean±std rows (multi-seed configs)."""
    by_group = {}
    for r in results:
        g = r.get("group", "")
        if not g:
            continue
        by_group.setdefault(g, []).append(r)

    agg_rows = []
    for group, items in by_group.items():
        means = np.array([r["ep_len_mean"] for r in items])
        succs = np.array([r["success_rate"] for r in items])
        algo = items[0]["algo"]
        reward = items[0]["reward"]
        agg_rows.append({
            "group": group,
            "algo": algo,
            "reward": reward,
            "n_seeds": len(items),
            "ep_len_mean_of_means": float(means.mean()),
            "ep_len_std_of_means": float(means.std(ddof=1)) if len(items) > 1 else 0.0,
            "ep_len_min_of_means": float(means.min()),
            "ep_len_max_of_means": float(means.max()),
            "success_rate_mean": float(succs.mean()),
        })

    csv_path = out_dir / "results_aggregate_v2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)
    print(f"[OK] {csv_path}")
    return agg_rows


def save_figure(fig, out_dir: Path, name: str) -> None:
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"[OK] {pdf_path}")
    print(f"[OK] {png_path}")


def plot_learning_curves(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for run in RUNS:
        data = read_tb_scalar(run.tb_dir, "rollout/ep_len_mean")
        if data is None:
            print(f"[skip] {run.name}: no ep_len_mean tag")
            continue
        steps, values = data
        # Multi-seed runs (seed 1, 2) plotted with same color but dashed
        ls = "--" if any(s in run.name for s in ["seed1", "seed2"]) else "-"
        lw = 1.2 if ls == "--" else 1.6
        ax1.plot(steps, smooth(values, window=5),
                 label=run.label, color=run.color, lw=lw, ls=ls)
    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Mean episode length")
    ax1.set_title("Episode length during training (10 runs)")
    ax1.set_ylim(0, 1050)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc="upper right", ncol=1)

    for run in RUNS:
        data = read_tb_scalar(run.tb_dir, "rollout/ep_rew_mean")
        if data is None:
            continue
        steps, values = data
        ls = "--" if any(s in run.name for s in ["seed1", "seed2"]) else "-"
        lw = 1.2 if ls == "--" else 1.6
        ax2.plot(steps, smooth(values, window=5),
                 label=run.label, color=run.color, lw=lw, ls=ls)
    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel("Mean episode return (training reward)")
    ax2.set_title("Training reward (shaping-dependent scale)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc="lower right")

    save_figure(fig, out_dir, "learning_curves_v2")
    plt.close(fig)


def plot_eval_summary(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    labels = [r["label"] for r in results]
    means = [r["ep_len_mean"] for r in results]
    stds = [r["ep_len_std"] for r in results]
    colors = [r["color"] for r in results]
    x = np.arange(len(results))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", lw=0.6, capsize=4)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Mean episode length")
    ax.set_title(
        f"Deterministic evaluation on sparse reward "
        f"(n = {results[0]['n_episodes']} episodes per run, 10 runs)"
    )
    ax.set_ylim(0, 1150)
    ax.axhline(1000, ls="--", color="grey", lw=0.7, alpha=0.7)
    ax.text(len(results) - 0.5, 1015, "TimeLimit", fontsize=8, color="grey")
    ax.grid(True, axis="y", alpha=0.3)
    for i, r in enumerate(results):
        ax.text(i, means[i] + stds[i] + 25,
                f"{r['success_rate']*100:.0f}%",
                ha="center", fontsize=8, fontweight="bold")
    save_figure(fig, out_dir, "eval_summary_v2")
    plt.close(fig)


def plot_multiseed_bands(out_dir: Path) -> None:
    """Show PPO+PBRS (3 seeds) and DQN sparse (2 seeds) as mean ± std bands."""
    groups = {
        "ppo_pbrs": {
            "label": "PPO — PBRS (n=3 seeds)",
            "color": "#9467bd",
            "runs": [r for r in RUNS if r.group == "ppo_pbrs"],
        },
        "dqn_sparse": {
            "label": "DQN — sparse (n=2 seeds)",
            "color": "#d62728",
            "runs": [r for r in RUNS if r.group == "dqn_sparse"],
        },
    }
    # Add bignet as its own line for context
    bignet = next((r for r in RUNS if r.name == "ppo_pbrs_bignet"), None)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

    for gname, gdata in groups.items():
        all_steps = []
        all_values = []
        for run in gdata["runs"]:
            data = read_tb_scalar(run.tb_dir, "rollout/ep_len_mean")
            if data is None:
                continue
            steps, values = data
            all_steps.append(steps)
            all_values.append(values)
        if not all_values:
            continue
        # Re-align all runs onto a common step grid (the shortest run dictates length)
        min_len = min(len(v) for v in all_values)
        common_steps = all_steps[0][:min_len]
        stacked = np.array([smooth(v[:min_len], window=5) for v in all_values])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(common_steps, mean, label=gdata["label"], color=gdata["color"], lw=2.0)
        ax.fill_between(common_steps, mean - std, mean + std,
                        color=gdata["color"], alpha=0.2)

    if bignet is not None:
        data = read_tb_scalar(bignet.tb_dir, "rollout/ep_len_mean")
        if data is not None:
            steps, values = data
            ax.plot(steps, smooth(values, window=5),
                    label="PPO — PBRS bignet [256,256] (n=1)",
                    color="#e377c2", lw=1.6, ls=":")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode length (smoothed window=5)")
    ax.set_title("Training curves with multi-seed confidence bands")
    ax.set_ylim(0, 1050)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    save_figure(fig, out_dir, "multiseed_curves_v2")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip eval and only regenerate plots from existing data.")
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
            "label": run.label, "color": run.color, "group": run.group, **stats,
        })
        print(f"     ep_len = {stats['ep_len_mean']:.1f} ± {stats['ep_len_std']:.1f}, "
              f"success = {stats['success_rate']*100:.0f}%")

    print("\n=== Generating outputs ===")
    if not args.skip_eval:
        write_table(results, out_dir)
        agg = write_aggregate(results, out_dir)
        print("\n=== Aggregate (multi-seed) ===")
        for r in agg:
            print(f"  {r['algo']} {r['reward']:6s} | n={r['n_seeds']} | "
                  f"ep_len = {r['ep_len_mean_of_means']:.1f} ± "
                  f"{r['ep_len_std_of_means']:.1f} | "
                  f"success = {r['success_rate_mean']*100:.0f}%")
        plot_eval_summary(results, out_dir)
    plot_learning_curves(out_dir)
    plot_multiseed_bands(out_dir)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
