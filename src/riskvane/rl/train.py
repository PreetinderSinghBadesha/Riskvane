"""Train tabular Q-learning and DQN remediation agents across seeds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

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
from riskvane.rl.env import RemediationEnv
from riskvane.rl.qlearning import (
    DEFAULT_ALPHA,
    DEFAULT_EPISODES,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_MIN,
    QLearningAgent,
)

DEFAULT_OUTPUT_DIR = Path("var/rl")
DEFAULT_SEED = 0
DEFAULT_SEEDS = 5


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _final_mean(history: list[float]) -> float:
    tail = history[-100:] if history else [0.0]
    return float(np.mean(tail))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskvane-train-rl",
        description="Train tabular Q-learning and stable-baselines3 DQN remediation agents.",
    )
    parser.add_argument("--agent", choices=("qlearning", "dqn", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reward-source", choices=("rules", "model"), default="rules")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--epsilon-min", type=float, default=DEFAULT_EPSILON_MIN)
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPSILON_DECAY)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-starts", type=int, default=DEFAULT_LEARNING_STARTS)
    parser.add_argument(
        "--target-update-interval", type=int, default=DEFAULT_TARGET_UPDATE_INTERVAL
    )
    parser.add_argument("--exploration-fraction", type=float, default=DEFAULT_EXPLORATION_FRACTION)
    parser.add_argument(
        "--exploration-initial-eps", type=float, default=DEFAULT_EXPLORATION_INITIAL_EPS
    )
    parser.add_argument(
        "--exploration-final-eps", type=float, default=DEFAULT_EXPLORATION_FINAL_EPS
    )
    parser.add_argument("--net-arch", type=int, nargs="+", default=DEFAULT_NET_ARCH)
    return parser


def _env(args, reward_source: str) -> RemediationEnv:
    return RemediationEnv(
        reward_source=reward_source,
        model_path=Path(args.model_path) if args.model_path else None,
        features_path=Path(args.features_path) if args.features_path else None,
    )


def _train_qlearning(args, seeds: list[int], output_dir: Path) -> list[dict]:
    metrics = []
    for seed in seeds:
        env = _env(args, args.reward_source)
        agent = QLearningAgent(
            n_budgets=env.budget + 1,
            costs=env.action_costs,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            seed=seed,
        )
        started = time.perf_counter()
        history = agent.train(env, episodes=args.episodes, seed=seed)
        seconds = time.perf_counter() - started
        agent.save(output_dir / f"q_table_seed{seed}.json")
        if seed == seeds[0]:
            agent.save(output_dir / "q_table.json")
        _write_json(
            output_dir / f"qlearning_history_seed{seed}.json",
            {
                "rewards": history,
                "episodes": args.episodes,
                "epsilon": agent.epsilon,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "seed": seed,
            },
        )
        metrics.append(
            {
                "seed": seed,
                "episodes": args.episodes,
                "seconds": seconds,
                "final_mean_reward": _final_mean(history),
            }
        )
    return metrics


def _train_dqn(args, seeds: list[int], output_dir: Path) -> list[dict]:
    metrics = []
    for seed in seeds:
        env = _env(args, args.reward_source)
        started = time.perf_counter()
        agent, history = DQNAgent.train(
            env,
            timesteps=args.timesteps,
            seed=seed,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_starts=args.learning_starts,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            net_arch=args.net_arch,
        )
        seconds = time.perf_counter() - started
        agent.save(output_dir / f"dqn_model_seed{seed}.zip")
        if seed == seeds[0]:
            agent.save(output_dir / "dqn_model.zip")
        _write_json(
            output_dir / f"dqn_history_seed{seed}.json",
            {
                "rewards": history,
                "timesteps": args.timesteps,
                "seed": seed,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
            },
        )
        metrics.append(
            {
                "seed": seed,
                "timesteps": args.timesteps,
                "seconds": seconds,
                "final_mean_reward": _final_mean(history),
            }
        )
    return metrics


def _print_metrics(metrics: dict[str, list[dict]]) -> None:
    header = f"{'Agent':<12} {'Seed':>6} {'Budget':>10} {'Seconds':>12} {'Final Mean Reward':>18}"
    print(header)
    print("-" * len(header))
    for agent_name, rows in metrics.items():
        for row in rows:
            budget = row.get("episodes", row.get("timesteps"))
            print(
                f"{agent_name:<12} {row['seed']:>6} {budget:>10} {row['seconds']:>12.2f} "
                f"{row['final_mean_reward']:>18.3f}"
            )
    print()
    header = f"{'Agent':<12} {'Mean Final Reward':>18} {'Std Final Reward':>18} {'Total Seconds':>14}"
    print(header)
    print("-" * len(header))
    for agent_name, rows in metrics.items():
        rewards = [row["final_mean_reward"] for row in rows]
        seconds = sum(row["seconds"] for row in rows)
        print(
            f"{agent_name:<12} {np.mean(rewards):>18.3f} {np.std(rewards):>18.3f} "
            f"{seconds:>14.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.seeds))

    metrics: dict[str, list[dict]] = {}
    if args.agent in ("qlearning", "both"):
        metrics["qlearning"] = _train_qlearning(args, seeds, output_dir)
    if args.agent in ("dqn", "both"):
        metrics["dqn"] = _train_dqn(args, seeds, output_dir)

    _write_json(output_dir / "training_metrics.json", metrics)
    _print_metrics(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
