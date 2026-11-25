"""Draw the result charts shown in README.md into docs/images/.

Reads the evaluation output in var/rl/ (run `riskvane-train-rl` and `riskvane-eval-rl` first)
and the trained XGBoost model in var/models/.

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from riskvane.posture.training.synthetic import DIMENSIONS, build_facts, generate_profiles
from riskvane.rl.baselines import OptimalPolicy
from riskvane.rl.env import RemediationEnv
from riskvane.rl.evaluate import run_episode
from riskvane.rl.planner import RemediationPlanner

RESULTS_DIR = Path("var/rl")
OUT_DIR = Path("docs/images")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

POLICY_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "cost_greedy": "Cost-greedy",
    "optimal": "Optimal (DP)",
    "qlearning": "Q-learning",
    "dqn": "DQN",
}
DIMENSION_LABELS = {
    "root_login": "Root SSH login",
    "password_auth": "Password auth",
    "cve_exposure": "CVE exposure",
    "unauthenticated_service": "Unauth. service",
    "world_writable_files": "World-writable files",
    "suid_binaries": "SUID binaries",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    }
)


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / name}")


def _load_results() -> dict[str, dict[str, dict]]:
    by_source: dict[str, dict[str, dict]] = {}
    with (RESULTS_DIR / "results.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_source.setdefault(row["reward_source"], {})[row["policy"]] = {
                key: float(value)
                for key, value in row.items()
                if key not in ("reward_source", "policy")
            }
    return by_source


def policy_rewards(results) -> None:
    policies = ["optimal", "cost_greedy", "greedy", "dqn", "qlearning", "random"]
    y = np.arange(len(policies))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for offset, source, color, label in (
        (-height / 2, "rules", BLUE, "Rule-based reward"),
        (height / 2, "model", ORANGE, "XGBoost reward"),
    ):
        values = [results[source][p]["mean_reward"] for p in policies]
        ax.barh(y + offset, values, height=height - 0.04, color=color, label=label, zorder=2)
        for yi, value in zip(y + offset, values, strict=True):
            ax.text(value + 0.4, yi, f"{value:.1f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [POLICY_LABELS[p] for p in policies])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean episode reward (63 start profiles × 200 episodes)")
    ax.set_title("Mean reward per remediation policy")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 40)
    _save(fig, "rl_policy_rewards.png")


def gap_to_optimal(results) -> None:
    rules = results["rules"]
    best = rules["optimal"]["mean_reward"]
    policies = ["cost_greedy", "greedy", "dqn", "qlearning", "random"]
    gaps = [100 * (best - rules[p]["mean_reward"]) / best for p in policies]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = [BLUE if p in ("dqn", "qlearning") else "#9b9a95" for p in policies]
    ax.barh(range(len(policies)), gaps, color=colors, height=0.6, zorder=2)
    for i, gap in enumerate(gaps):
        ax.text(gap + 1, i, f"{gap:.1f}%", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(policies)), [POLICY_LABELS[p] for p in policies])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Reward shortfall vs. the DP-optimal policy (%, rule-based reward)")
    ax.set_title("How far each policy falls short of optimal")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color="#9b9a95"),
    ]
    ax.legend(handles, ["Learned agent", "Baseline"], loc="center right", fontsize=9)
    _save(fig, "rl_gap_to_optimal.png")


def _seed_histories(prefix: str) -> list[np.ndarray]:
    histories = []
    for path in sorted(RESULTS_DIR.glob(f"{prefix}_history_seed*.json")):
        histories.append(np.asarray(json.loads(path.read_text(encoding="utf-8"))["rewards"], float))
    return histories


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def learning_curves(results) -> None:
    optimal = results["rules"]["optimal"]["mean_reward"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout="constrained")
    for ax, (prefix, title, window) in zip(
        axes,
        (("qlearning", "Q-learning (5 seeds)", 1000), ("dqn", "DQN (3 seeds)", 200)),
        strict=True,
    ):
        histories = _seed_histories(prefix)
        length = min(len(h) for h in histories)
        curves = np.stack([_rolling(h[:length], window) for h in histories])
        x = np.arange(curves.shape[1]) + window
        ax.fill_between(
            x, curves.min(0), curves.max(0), color=BLUE, alpha=0.18, lw=0, label="Seed range"
        )
        ax.plot(x, curves.mean(0), color=BLUE, lw=2, label="Mean across seeds")
        ax.axhline(optimal, color=INK_2, lw=1.2, ls="--", label=f"Optimal policy ({optimal:.1f})")
        ax.set_title(title)
        ax.set_xlabel("Training episode")
    axes[0].set_ylabel("Episode reward (rolling mean)")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Training curves (rule-based reward)", x=0.01, ha="left", fontsize=13, weight="bold"
    )
    _save(fig, "rl_learning_curves.png")


def score_model_vs_rules() -> None:
    rules_env = RemediationEnv(reward_source="rules")
    model_env = RemediationEnv(reward_source="model")
    profiles = generate_profiles()
    states = [np.array([int(p[name]) for name in DIMENSIONS]) for p in profiles]
    rule_scores = np.array([rules_env.score_state(s) for s in states])
    model_scores = np.array([model_env.score_state(s) for s in states])
    mae = float(np.mean(np.abs(rule_scores - model_scores)))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 100], [0, 100], color=INK_2, lw=1, ls="--", label="Perfect agreement", zorder=1)
    ax.scatter(
        rule_scores,
        model_scores,
        s=44,
        color=BLUE,
        edgecolor=SURFACE,
        lw=1.5,
        zorder=2,
        label="Synthetic profile (64)",
    )
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Rule-based risk score")
    ax.set_ylabel("XGBoost predicted risk score")
    ax.set_title(f"XGBoost vs. rule-based score (MAE {mae:.1f})")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "model_vs_rules.png")


def success_by_profile() -> None:
    env = RemediationEnv(reward_source="rules")
    policy = OptimalPolicy(env)
    groups = {
        "Neither": [],
        "Unauth. service only": [],
        "CVE only": [],
        "CVE + unauth. service": [],
    }
    for profile in generate_profiles():
        if not any(profile.values()):
            continue
        cve, unauth = profile["cve_exposure"], profile["unauthenticated_service"]
        key = (
            "CVE + unauth. service"
            if cve and unauth
            else "CVE only"
            if cve
            else "Unauth. service only"
            if unauth
            else "Neither"
        )
        for seed in range(200):
            groups[key].append(run_episode(env, policy, profile, seed=seed)["success"])
    names = list(groups)
    rates = [100 * float(np.mean(groups[n])) for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(names, rates, color=BLUE, width=0.6, zorder=2)
    for i, rate in enumerate(rates):
        ax.text(i, rate + 2, f"{rate:.1f}%", ha="center", fontsize=9, color=INK_2)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Episodes ending fully clean (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Optimal-policy success rate by hard-to-fix findings present")
    _save(fig, "rl_success_by_profile.png")


def remediation_trajectory() -> None:
    profile = {name: name != "cve_exposure" for name in DIMENSIONS}
    plan = RemediationPlanner().plan(build_facts("demo-vm", profile), policy="optimal")
    spent = [0.0]
    scores = [plan.initial_score]
    for step in plan.steps:
        spent.append(spent[-1] + step.cost)
        scores.append(step.score_after)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.step(spent, scores, where="post", color=BLUE, lw=2, zorder=2)
    ax.plot(spent, scores, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, zorder=3)
    for x, score, step in zip(spent[1:], scores[1:], plan.steps, strict=True):
        ax.annotate(
            f"{step.order}. {DIMENSION_LABELS[step.dimension]}\n→ {score}",
            (x, score),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_xlabel("Remediation budget spent")
    ax.set_ylabel("Risk score after the fix")
    ax.set_ylim(-5, 115)
    ax.set_xlim(-0.3, plan.budget_spent + 1.6)
    ax.set_title("POST /remediate plan (optimal policy) for a VM with 5 misconfigurations")
    ax.text(
        0,
        -0.2,
        "The risk score is capped at 100, so the first fix shows no drop.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_2,
        va="top",
    )
    _save(fig, "remediate_trajectory.png")


def main() -> None:
    results = _load_results()
    policy_rewards(results)
    gap_to_optimal(results)
    learning_curves(results)
    success_by_profile()
    score_model_vs_rules()
    remediation_trajectory()


if __name__ == "__main__":
    main()
