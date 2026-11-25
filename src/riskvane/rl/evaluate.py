"""Evaluate remediation policies on the full non-clean profile space."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from riskvane.posture.training.synthetic import DIMENSIONS, generate_profiles
from riskvane.rl.baselines import CostGreedyPolicy, GreedyPolicy, OptimalPolicy, RandomPolicy
from riskvane.rl.dqn import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_EXPLORATION_FINAL_EPS,
    DEFAULT_EXPLORATION_FRACTION,
    DEFAULT_EXPLORATION_INITIAL_EPS,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LEARNING_STARTS,
    DEFAULT_NET_ARCH,
    DEFAULT_TARGET_UPDATE_INTERVAL,
    DEFAULT_TIMESTEPS,
    DQNAgent,
)
from riskvane.rl.env import (
    ACTION_COSTS,
    DEFAULT_BUDGET,
    DEFAULT_COMPLETION_BONUS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NOOP_PENALTY,
    DEFAULT_SIDE_EFFECT_PENALTY,
    DEFAULT_SIDE_EFFECT_PROB,
    SIDE_EFFECT_ACTION,
    SIDE_EFFECT_RESET,
    SUCCESS_PROBABILITIES,
    RemediationEnv,
)
from riskvane.rl.qlearning import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_MIN,
    QLearningAgent,
)
from riskvane.rl.qlearning import (
    DEFAULT_EPISODES as QLEARNING_EPISODES,
)
from riskvane.rl.qlearning import (
    DEFAULT_GAMMA as Q_DEFAULT_GAMMA,
)

DEFAULT_OUTPUT_DIR = Path("var/rl")
DEFAULT_EPISODES = 200
DEFAULT_SEED = 0
DEFAULT_NUM_SEEDS = 5
DEFAULT_LATENCY_CALLS = 10000
POLICY_ORDER = ("random", "greedy", "cost_greedy", "optimal", "qlearning", "dqn")


def profile_label(profile: dict[str, bool]) -> str:
    return "".join("1" if profile[name] else "0" for name in DIMENSIONS)


def run_episode(env: RemediationEnv, policy, profile: dict[str, bool], seed: int) -> dict:
    if hasattr(policy, "reset"):
        policy.reset(seed=seed)
    observation, info = env.reset(seed=seed, options={"profile": profile})
    if env.is_clean(observation):
        return {"reward": 0.0, "steps": 0, "success": True, "cost": 0.0}

    total_reward = 0.0
    total_cost = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cost += float(info["cost"])
        steps += 1

    return {
        "reward": total_reward,
        "steps": steps,
        "success": env.is_clean(observation),
        "cost": total_cost,
    }


def _profile_records(profile, episodes: int, results: list[dict]) -> dict:
    rewards = [result["reward"] for result in results]
    steps = [result["steps"] for result in results]
    successes = [1.0 if result["success"] else 0.0 for result in results]
    costs = [result["cost"] for result in results]
    return {
        "profile": profile_label(profile),
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(successes)),
        "mean_cost": float(np.mean(costs)),
    }


def _aggregate(name: str, seed: int | None, results: list[dict], gaps: list[float]) -> dict:
    rewards = [result["reward"] for result in results]
    steps = [result["steps"] for result in results]
    successes = [1.0 if result["success"] else 0.0 for result in results]
    costs = [result["cost"] for result in results]
    return {
        "policy": name,
        "seed": seed,
        "episodes": len(results),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "success_rate": float(np.mean(successes)),
        "std_success_rate": float(np.std(successes)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "gap_to_optimal": float(np.mean(gaps)),
        "std_gap_to_optimal": float(np.std(gaps)),
    }


def evaluate_policy(
    env: RemediationEnv,
    policy,
    profiles: list[dict[str, bool]],
    episodes: int,
    seed: int,
    optimal_rewards: dict[tuple[int, int], float] | None = None,
) -> dict:
    all_results: list[dict] = []
    all_gaps: list[float] = []
    profile_records: list[dict] = []

    for profile_index, profile in enumerate(profiles):
        results = []
        for episode in range(episodes):
            episode_seed = seed + profile_index * episodes + episode
            result = run_episode(env, policy, profile, episode_seed)
            results.append(result)
            all_results.append(result)
            if optimal_rewards is not None:
                all_gaps.append(optimal_rewards[(profile_index, episode)] - result["reward"])
        profile_records.append(_profile_records(profile, episodes, results))

    aggregate = _aggregate(policy.name, getattr(policy, "seed", None), all_results, all_gaps)
    aggregate["profiles"] = profile_records
    return aggregate


def evaluate_optimal(
    env: RemediationEnv,
    policy: OptimalPolicy,
    profiles: list[dict[str, bool]],
    episodes: int,
    seed: int,
) -> tuple[dict, dict[tuple[int, int], float]]:
    all_results: list[dict] = []
    profile_records: list[dict] = []
    rewards: dict[tuple[int, int], float] = {}

    for profile_index, profile in enumerate(profiles):
        results = []
        for episode in range(episodes):
            episode_seed = seed + profile_index * episodes + episode
            result = run_episode(env, policy, profile, episode_seed)
            results.append(result)
            all_results.append(result)
            rewards[(profile_index, episode)] = result["reward"]
        profile_records.append(_profile_records(profile, episodes, results))

    aggregate = _aggregate("optimal", None, all_results, [0.0] * len(all_results))
    aggregate["profiles"] = profile_records
    return aggregate, rewards


def _group_summary(policy_results: list[dict]) -> list[dict]:
    summary = []
    for name in dict.fromkeys(result["policy"] for result in policy_results):
        group = [result for result in policy_results if result["policy"] == name]
        if len(group) == 1:
            result = group[0]
            summary.append(
                {
                    "policy": name,
                    "seeds": 1,
                    "mean_reward": result["mean_reward"],
                    "std_reward": result["std_reward"],
                    "mean_steps": result["mean_steps"],
                    "std_steps": result["std_steps"],
                    "success_rate": result["success_rate"],
                    "std_success_rate": result["std_success_rate"],
                    "mean_cost": result["mean_cost"],
                    "std_cost": result["std_cost"],
                    "gap_to_optimal": result["gap_to_optimal"],
                    "std_gap_to_optimal": result["std_gap_to_optimal"],
                }
            )
        else:
            summary.append(
                {
                    "policy": name,
                    "seeds": len(group),
                    "mean_reward": float(np.mean([item["mean_reward"] for item in group])),
                    "std_reward": float(np.std([item["mean_reward"] for item in group])),
                    "mean_steps": float(np.mean([item["mean_steps"] for item in group])),
                    "std_steps": float(np.std([item["mean_steps"] for item in group])),
                    "success_rate": float(np.mean([item["success_rate"] for item in group])),
                    "std_success_rate": float(
                        np.std([item["success_rate"] for item in group])
                    ),
                    "mean_cost": float(np.mean([item["mean_cost"] for item in group])),
                    "std_cost": float(np.std([item["mean_cost"] for item in group])),
                    "gap_to_optimal": float(
                        np.mean([item["gap_to_optimal"] for item in group])
                    ),
                    "std_gap_to_optimal": float(
                        np.std([item["gap_to_optimal"] for item in group])
                    ),
                }
            )
    summary.sort(key=lambda row: POLICY_ORDER.index(row["policy"]))
    return summary


def evaluate_source(
    env: RemediationEnv,
    policies: list,
    profiles: list[dict[str, bool]],
    episodes: int,
    seed: int,
) -> dict:
    optimal_policy = next(policy for policy in policies if policy.name == "optimal")
    optimal_result, optimal_rewards = evaluate_optimal(env, optimal_policy, profiles, episodes, seed)
    policy_results = [optimal_result]
    for policy in policies:
        if policy.name == "optimal":
            continue
        policy_results.append(
            evaluate_policy(env, policy, profiles, episodes, seed, optimal_rewards)
        )
    return {
        "summary": _group_summary(policy_results),
        "policies": policy_results,
    }


def _seed_paths(
    output_dir: Path, pattern: str, fallback: str, seed: int, num_seeds: int
) -> list[tuple[int, Path]]:
    """Per-seed artifacts that exist; the unsuffixed file only when no seeded file exists.

    Falling back per missing seed would count one artifact several times, or mix in a
    stale artifact from an older environment version.
    """
    found = [
        (current, output_dir / pattern.format(current))
        for current in range(seed, seed + num_seeds)
        if (output_dir / pattern.format(current)).exists()
    ]
    if not found and (output_dir / fallback).exists():
        found = [(seed, output_dir / fallback)]
    return found


def _load_seed_agents(
    env: RemediationEnv,
    output_dir: Path,
    seed: int,
    num_seeds: int,
) -> tuple[list[QLearningAgent], list[DQNAgent]]:
    q_paths = _seed_paths(output_dir, "q_table_seed{}.json", "q_table.json", seed, num_seeds)
    dqn_paths = _seed_paths(output_dir, "dqn_model_seed{}.zip", "dqn_model.zip", seed, num_seeds)
    q_agents = [QLearningAgent.load(path) for _, path in q_paths]
    dqn_agents = [DQNAgent.load(path, env=env, seed=current) for current, path in dqn_paths]
    return q_agents, dqn_agents


def _build_policies(env: RemediationEnv, output_dir: Path, seed: int, num_seeds: int) -> list:
    policies = [
        RandomPolicy(env.action_space.n, seed=seed),
        GreedyPolicy(env),
        CostGreedyPolicy(env),
        OptimalPolicy(env),
    ]
    q_agents, dqn_agents = _load_seed_agents(env, output_dir, seed, num_seeds)
    policies.extend(q_agents)
    policies.extend(dqn_agents)
    return policies


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _load_histories(output_dir: Path, key: str, seed: int, num_seeds: int) -> list[list[float]]:
    histories = []
    paths = _seed_paths(
        output_dir, f"{key}_history_seed{{}}.json", f"{key}_history.json", seed, num_seeds
    )
    for _, path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewards = payload.get("rewards", [])
        if rewards:
            histories.append([float(value) for value in rewards])
    return histories


def plot_learning_curves(
    output_dir: Path,
    seed: int,
    num_seeds: int,
    optimal_level: float,
    greedy_level: float,
) -> None:
    curves = (
        ("qlearning", "Q-learning learning curve"),
        ("dqn", "DQN learning curve"),
    )
    for key, title in curves:
        histories = _load_histories(output_dir, key, seed, num_seeds)
        if not histories:
            continue
        length = min(len(history) for history in histories)
        if length == 0:
            continue
        matrix = np.asarray([history[:length] for history in histories], dtype=float)
        window = max(1, length // 100)
        smoothed = np.asarray([_rolling_mean(row, window) for row in matrix])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        episodes = np.arange(1, length + 1)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.fill_between(episodes, mean - std, mean + std, color="#c0392b", alpha=0.2)
        axis.plot(episodes, mean, color="#c0392b", linewidth=2, label="Mean")
        axis.axhline(optimal_level, color="#27ae60", linestyle="--", label="Optimal")
        axis.axhline(greedy_level, color="#8e44ad", linestyle=":", label="Greedy")
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel("Cumulative reward")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{key}_learning_curve.png", dpi=150)
        plt.close(figure)


def _measure_latency(policy, seed: int, calls: int = DEFAULT_LATENCY_CALLS) -> float:
    rng = np.random.default_rng(seed)
    observations = []
    for _ in range(calls):
        bits = int(rng.integers(0, 1 << len(DIMENSIONS)))
        budget = int(rng.integers(0, DEFAULT_BUDGET + 1))
        observations.append(
            np.array([(bits >> index) & 1 for index in range(len(DIMENSIONS))] + [budget], dtype=np.int8)
        )
    for observation in observations[:100]:
        policy.act(observation)
    start = time.perf_counter()
    for observation in observations:
        policy.act(observation)
    elapsed = time.perf_counter() - start
    return float(elapsed / calls * 1000.0)


def _hardware() -> dict:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except (ImportError, RuntimeError):
        gpu = "none"
    return {
        "cpu": platform.processor() or platform.uname().processor,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "gpu": gpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _read_training_metrics(output_dir: Path) -> dict:
    path = output_dir / "training_metrics.json"
    if not path.exists():
        return {"qlearning": [], "dqn": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_results(output_dir: Path, sources: dict, config: dict) -> None:
    fields = [
        "reward_source",
        "policy",
        "seeds",
        "mean_reward",
        "std_reward",
        "mean_steps",
        "std_steps",
        "success_rate",
        "std_success_rate",
        "mean_cost",
        "std_cost",
        "gap_to_optimal",
        "std_gap_to_optimal",
    ]
    rows = []
    for source_name, source in sources.items():
        for row in source["summary"]:
            rows.append({"reward_source": source_name, **row})
    with open(output_dir / "results.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"config": config, "sources": sources}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_metric(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def _print_summary(summary: list[dict], title: str) -> None:
    header = (
        f"{title}\n"
        f"{'Policy':<14} {'Mean Reward':>20} {'Mean Steps':>20} {'Success Rate':>20} "
        f"{'Mean Cost':>20} {'Gap to Optimal':>20}"
    )
    print(header)
    print("-" * len(header.splitlines()[-1]))
    for row in summary:
        print(
            f"{row['policy']:<14} "
            f"{_format_metric(row['mean_reward'], row['std_reward']):>20} "
            f"{_format_metric(row['mean_steps'], row['std_steps']):>20} "
            f"{_format_metric(row['success_rate'], row['std_success_rate']):>20} "
            f"{_format_metric(row['mean_cost'], row['std_cost']):>20} "
            f"{_format_metric(row['gap_to_optimal'], row['std_gap_to_optimal']):>20}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskvane-eval-rl",
        description="Evaluate remediation policies on all non-clean profiles.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--reward-source", choices=("rules", "model"), default="rules")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--features-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [profile for profile in generate_profiles() if any(profile.values())]

    envs = {
        "rules": RemediationEnv(reward_source="rules"),
        "model": RemediationEnv(
            reward_source="model",
            model_path=Path(args.model_path) if args.model_path else None,
            features_path=Path(args.features_path) if args.features_path else None,
        ),
    }

    sources = {}
    for source_name in ("rules", "model"):
        env = envs[source_name]
        policies = _build_policies(env, output_dir, args.seed, args.num_seeds)
        sources[source_name] = evaluate_source(
            env, policies, profiles, args.episodes, args.seed
        )

    rules_summary = {row["policy"]: row for row in sources["rules"]["summary"]}
    plot_learning_curves(
        output_dir,
        args.seed,
        args.num_seeds,
        rules_summary["optimal"]["mean_reward"],
        rules_summary["greedy"]["mean_reward"],
    )

    rules_policies = _build_policies(envs["rules"], output_dir, args.seed, args.num_seeds)
    q_policy = next(policy for policy in rules_policies if policy.name == "qlearning")
    dqn_policy = next(policy for policy in rules_policies if policy.name == "dqn")
    latency = {
        "qlearning": _measure_latency(q_policy, args.seed),
        "dqn": _measure_latency(dqn_policy, args.seed),
    }

    config = {
        "episodes": args.episodes,
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "profiles": len(profiles),
        "reward_sources": ["rules", "model"],
        "primary_reward_source": args.reward_source,
    }
    _write_results(output_dir, sources, config)

    report_data = {
        "environment": {
            "dimensions": list(DIMENSIONS),
            "action_costs": ACTION_COSTS,
            "success_probabilities": SUCCESS_PROBABILITIES,
            "side_effect": {
                "action": SIDE_EFFECT_ACTION,
                "probability": DEFAULT_SIDE_EFFECT_PROB,
                "penalty": DEFAULT_SIDE_EFFECT_PENALTY,
                "resets": SIDE_EFFECT_RESET,
            },
            "budget": DEFAULT_BUDGET,
            "max_steps": DEFAULT_MAX_STEPS,
            "noop_penalty": DEFAULT_NOOP_PENALTY,
            "completion_bonus": DEFAULT_COMPLETION_BONUS,
            "start_state_rule": "uniform over 63 non-clean profiles from the 64 factorial profiles",
            "budget_exhaustion_rule": "remaining budget cannot cover any remaining misconfiguration",
        },
        "qlearning": {
            "alpha": DEFAULT_ALPHA,
            "gamma": Q_DEFAULT_GAMMA,
            "epsilon_start": DEFAULT_EPSILON,
            "epsilon_end": DEFAULT_EPSILON_MIN,
            "epsilon_decay": DEFAULT_EPSILON_DECAY,
            "episodes": QLEARNING_EPISODES,
            "n_states": 64 * (DEFAULT_BUDGET + 1),
            "n_budgets": DEFAULT_BUDGET + 1,
            "seeds": list(range(args.seed, args.seed + args.num_seeds)),
        },
        "dqn": {
            "learning_rate": DEFAULT_LEARNING_RATE,
            "buffer_size": DEFAULT_BUFFER_SIZE,
            "batch_size": DEFAULT_BATCH_SIZE,
            "network_architecture": DEFAULT_NET_ARCH,
            "gamma": DEFAULT_GAMMA,
            "learning_starts": DEFAULT_LEARNING_STARTS,
            "exploration_fraction": DEFAULT_EXPLORATION_FRACTION,
            "exploration_initial_eps": DEFAULT_EXPLORATION_INITIAL_EPS,
            "exploration_final_eps": DEFAULT_EXPLORATION_FINAL_EPS,
            "target_update_interval": DEFAULT_TARGET_UPDATE_INTERVAL,
            "total_timesteps": DEFAULT_TIMESTEPS,
            "seeds": list(range(args.seed, args.seed + args.num_seeds)),
        },
        "training": _read_training_metrics(output_dir),
        "hardware": _hardware(),
        "inference_latency_ms": latency,
        "results": {
            "rules": sources["rules"]["summary"],
            "model": sources["model"]["summary"],
        },
        "artifacts": {
            "results_csv": str(output_dir / "results.csv"),
            "results_json": str(output_dir / "results.json"),
            "qlearning_learning_curve": str(output_dir / "qlearning_learning_curve.png"),
            "dqn_learning_curve": str(output_dir / "dqn_learning_curve.png"),
        },
    }
    (output_dir / "report_data.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    _print_summary(sources["rules"]["summary"], "Rules reward source")
    _print_summary(sources["model"]["summary"], "Model reward source")
    print(json.dumps(report_data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
