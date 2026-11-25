"""Gymnasium environment for stochastic, budgeted VM remediation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from riskvane.posture.scoring import cve_feed, scorer
from riskvane.posture.training.features import featurize
from riskvane.posture.training.synthetic import DIMENSIONS, build_facts, generate_profiles

RewardSource = Literal["rules", "model"]

DEFAULT_BUDGET = 8
DEFAULT_MAX_STEPS = 12
DEFAULT_NOOP_PENALTY = 2.0
DEFAULT_COMPLETION_BONUS = 10.0
DEFAULT_SIDE_EFFECT_PROB = 0.3
DEFAULT_SIDE_EFFECT_PENALTY = 8.0

ACTION_COSTS = {
    "root_login": 1.0,
    "password_auth": 1.0,
    "cve_exposure": 2.0,
    "unauthenticated_service": 2.0,
    "world_writable_files": 1.0,
    "suid_binaries": 1.0,
}

SUCCESS_PROBABILITIES = {
    "root_login": 0.95,
    "password_auth": 0.95,
    "cve_exposure": 0.1,
    "unauthenticated_service": 0.2,
    "world_writable_files": 0.9,
    "suid_binaries": 0.9,
}

SIDE_EFFECT_ACTION = "unauthenticated_service"
SIDE_EFFECT_RESET = "world_writable_files"


class RemediationEnv(gym.Env):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        reward_source: RewardSource = "rules",
        model_path: Path | None = None,
        features_path: Path | None = None,
        budget: int = DEFAULT_BUDGET,
        max_steps: int = DEFAULT_MAX_STEPS,
        noop_penalty: float = DEFAULT_NOOP_PENALTY,
        completion_bonus: float = DEFAULT_COMPLETION_BONUS,
        side_effect_prob: float = DEFAULT_SIDE_EFFECT_PROB,
        side_effect_penalty: float = DEFAULT_SIDE_EFFECT_PENALTY,
    ):
        super().__init__()
        if reward_source not in ("rules", "model"):
            raise ValueError(f"unsupported reward source: {reward_source!r}")

        self.reward_source = reward_source
        self.budget = budget
        self.max_steps = max_steps
        self.noop_penalty = noop_penalty
        self.completion_bonus = completion_bonus
        self.side_effect_prob = side_effect_prob
        self.side_effect_penalty = side_effect_penalty

        self.dimensions = DIMENSIONS
        self.action_costs = [ACTION_COSTS[name] for name in DIMENSIONS]
        self.success_probs = [SUCCESS_PROBABILITIES[name] for name in DIMENSIONS]
        self.unauth_index = DIMENSIONS.index(SIDE_EFFECT_ACTION)
        self.wwf_index = DIMENSIONS.index(SIDE_EFFECT_RESET)

        self.action_space = spaces.Discrete(len(DIMENSIONS))
        self.observation_space = spaces.MultiDiscrete([2] * len(DIMENSIONS) + [budget + 1])

        self._profiles = generate_profiles()
        self._initial_profiles = [profile for profile in self._profiles if any(profile.values())]
        self._feed = cve_feed.load_feed()
        self._facts: dict[tuple[int, ...], Any] = {}
        self._score_cache: dict[tuple[int, ...], float] = {}
        self._model = self._load_model(model_path, features_path) if reward_source == "model" else None
        self._state = np.zeros(len(DIMENSIONS), dtype=np.int8)
        self._remaining_budget = budget
        self._elapsed_steps = 0
        self._current_score = 0.0

    @staticmethod
    def _load_model(model_path: Path | None, features_path: Path | None):
        from riskvane.posture.ml.inference import RiskModel

        kwargs = {}
        if model_path is not None:
            kwargs["model_path"] = Path(model_path)
        if features_path is not None:
            kwargs["features_path"] = Path(features_path)
        return RiskModel.load(**kwargs) if kwargs else RiskModel.load()

    @staticmethod
    def bits_index(state) -> int:
        return sum(1 << i for i, value in enumerate(state[: len(DIMENSIONS)]) if int(value))

    def state_index(self, observation) -> int:
        return self.bits_index(observation) * (self.budget + 1) + int(observation[len(DIMENSIONS)])

    def profile_to_state(self, profile: dict[str, bool]) -> np.ndarray:
        return np.array([int(bool(profile[name])) for name in self.dimensions], dtype=np.int8)

    def state_to_profile(self, state) -> dict[str, bool]:
        return {name: bool(int(state[i])) for i, name in enumerate(self.dimensions)}

    def _facts_for_state(self, key: tuple[int, ...]):
        if key not in self._facts:
            index = self.bits_index(key)
            self._facts[key] = build_facts(f"rl-{index:03d}", self.state_to_profile(key))
        return self._facts[key]

    def score_state(self, state) -> float:
        key = tuple(int(value) for value in state[: len(DIMENSIONS)])
        if key not in self._score_cache:
            facts = self._facts_for_state(key)
            if self._model is None:
                self._score_cache[key] = float(scorer.score(facts, self._feed).score)
            else:
                self._score_cache[key] = float(self._model.predict(featurize(facts))[0])
        return self._score_cache[key]

    def _observation(self) -> np.ndarray:
        return np.array([*self._state, self._remaining_budget], dtype=np.int8)

    def _info(self) -> dict[str, Any]:
        return {
            "score": self._current_score,
            "profile": self.state_to_profile(self._state),
            "budget": self._remaining_budget,
            "elapsed_steps": self._elapsed_steps,
        }

    def has_affordable_broken_action(self, state, budget: int) -> bool:
        return any(
            int(state[index]) == 1 and self.action_costs[index] <= budget
            for index in range(len(self.dimensions))
        )

    def is_clean(self, state) -> bool:
        """True when no risk dimension is set.

        Cleanliness is defined on the state bits, not the score: the trained model's
        prediction for a fully hardened VM is not exactly 0, so a score check would
        never fire under ``reward_source="model"``.
        """
        return not any(int(value) for value in state[: len(self.dimensions)])

    def is_terminal(self, state, budget: int, score: float | None = None) -> bool:
        return self.is_clean(state) or not self.has_affordable_broken_action(state, budget)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        if "state" in options:
            state = np.asarray(options["state"], dtype=np.int8)
            if state.shape[0] == len(self.dimensions) + 1:
                budget = int(state[len(self.dimensions)])
                state = state[: len(self.dimensions)]
            else:
                budget = int(options.get("budget", self.budget))
        elif "profile" in options:
            state = self.profile_to_state(options["profile"])
            budget = int(options.get("budget", self.budget))
        elif "profile_index" in options:
            state = self.profile_to_state(self._profiles[int(options["profile_index"])])
            budget = int(options.get("budget", self.budget))
        else:
            index = int(self.np_random.integers(0, len(self._initial_profiles)))
            state = self.profile_to_state(self._initial_profiles[index])
            budget = int(options.get("budget", self.budget))

        if state.shape != (len(self.dimensions),):
            raise ValueError("initial state does not match the observation space")
        if not 0 <= budget <= self.budget:
            raise ValueError("initial budget is outside the observation space")

        self._state = state.astype(np.int8, copy=True)
        self._remaining_budget = budget
        self._elapsed_steps = 0
        self._current_score = self.score_state(self._state)
        return self._observation(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        action = int(action)
        score_before = self._current_score
        cost = self.action_costs[action]
        noop = bool(self._state[action] == 0 or cost > self._remaining_budget)
        unaffordable = bool(cost > self._remaining_budget)
        success = False
        side_effect = False
        spent = 0.0

        if noop:
            reward = -self.noop_penalty
            score_after = score_before
        else:
            spent = cost
            self._remaining_budget -= int(cost)
            success = bool(self.np_random.random() < self.success_probs[action])

            if success:
                self._state[action] = 0
                if action == self.unauth_index and self.np_random.random() < self.side_effect_prob:
                    side_effect = True
                    self._state[self.wwf_index] = 1
                score_after = self.score_state(self._state)
                reward = score_before - score_after - cost
                if side_effect:
                    reward -= self.side_effect_penalty
                if self.is_clean(self._state):
                    reward += self.completion_bonus
            else:
                score_after = score_before
                reward = -cost

        self._elapsed_steps += 1
        self._current_score = score_after
        terminated = self.is_terminal(self._state, self._remaining_budget, score_after)
        truncated = not terminated and self._elapsed_steps >= self.max_steps

        info = self._info()
        info.update(
            {
                "action": action,
                "score_before": score_before,
                "score_after": score_after,
                "noop": noop,
                "unaffordable": unaffordable,
                "success": success,
                "side_effect": side_effect,
                "cost": spent,
            }
        )
        return self._observation(), float(reward), terminated, truncated, info
