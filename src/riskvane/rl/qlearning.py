"""Tabular Q-learning over the remediation state and budget space."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from riskvane.posture.training.synthetic import DIMENSIONS
from riskvane.rl.env import ACTION_COSTS, DEFAULT_BUDGET

DEFAULT_EPISODES = 50000
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.95
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_MIN = 0.02
DEFAULT_EPSILON_DECAY = 0.9998


class QLearningAgent:
    name = "qlearning"

    def __init__(
        self,
        n_states: int = 64 * (DEFAULT_BUDGET + 1),
        n_actions: int = len(DIMENSIONS),
        n_budgets: int = DEFAULT_BUDGET + 1,
        costs: list[float] | None = None,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
        epsilon_min: float = DEFAULT_EPSILON_MIN,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        seed: int = 0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_budgets = n_budgets
        self.costs = list(costs) if costs is not None else [ACTION_COSTS[name] for name in DIMENSIONS]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.q_table = np.zeros((n_states, n_actions), dtype=float)
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def bits_index(observation) -> int:
        return sum(1 << i for i, value in enumerate(observation[: len(DIMENSIONS)]) if int(value))

    def state_index(self, observation) -> int:
        return self.bits_index(observation) * self.n_budgets + int(observation[len(DIMENSIONS)])

    def valid_actions(self, observation) -> list[int]:
        state = observation[: len(DIMENSIONS)]
        budget = int(observation[len(DIMENSIONS)])
        return [
            action
            for action in range(self.n_actions)
            if int(state[action]) == 1 and self.costs[action] <= budget
        ]

    def act(self, observation, explore: bool = False) -> int:
        valid = self.valid_actions(observation)
        if not valid:
            return 0
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.choice(valid))
        index = self.state_index(observation)
        return int(valid[int(np.argmax(self.q_table[index, valid]))])

    def train(self, env, episodes: int = DEFAULT_EPISODES, seed: int | None = None) -> list[float]:
        seed = self.seed if seed is None else seed
        history: list[float] = []

        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                index = self.state_index(observation)
                action = self.act(observation, explore=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_index = self.state_index(next_observation)
                target = reward
                if not terminated:
                    valid_next = self.valid_actions(next_observation)
                    if valid_next:
                        target += self.gamma * float(np.max(self.q_table[next_index, valid_next]))
                self.q_table[index, action] += self.alpha * (target - self.q_table[index, action])
                observation = next_observation
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            history.append(float(total_reward))

        return history

    def greedy_action(self, observation) -> int:
        return self.act(observation, explore=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "n_budgets": self.n_budgets,
            "costs": self.costs,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "seed": self.seed,
            "q_table": self.q_table.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> QLearningAgent:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            n_states=payload["n_states"],
            n_actions=payload["n_actions"],
            n_budgets=payload["n_budgets"],
            costs=payload["costs"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            seed=payload["seed"],
        )
        agent.q_table = np.asarray(payload["q_table"], dtype=float)
        return agent
