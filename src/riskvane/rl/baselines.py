"""Reference policies for the remediation environment."""

from __future__ import annotations

import numpy as np

from riskvane.rl.env import RemediationEnv


class RandomPolicy:
    name = "random"

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation) -> int:
        return int(self.rng.integers(0, self.n_actions))


class GreedyPolicy:
    name = "greedy"

    def __init__(self, env: RemediationEnv):
        self.env = env
        self.weights = [
            env.score_state(np.eye(len(env.dimensions), dtype=np.int8)[index])
            for index in range(len(env.dimensions))
        ]

    def _score_drop(self, state, action: int) -> float:
        if int(state[action]) == 0:
            return 0.0
        next_state = state.copy()
        next_state[action] = 0
        return self.env.score_state(state) - self.env.score_state(next_state)

    def act(self, observation) -> int:
        state = np.asarray(observation[: len(self.env.dimensions)], dtype=np.int8)
        budget = int(observation[len(self.env.dimensions)])
        best_action = -1
        best_key = None
        for action in range(len(self.env.dimensions)):
            if int(state[action]) == 0 or self.env.action_costs[action] > budget:
                continue
            key = (self._score_drop(state, action), self.weights[action])
            if best_key is None or key > best_key:
                best_key = key
                best_action = action
        if best_action < 0:
            broken = np.flatnonzero(state == 1)
            return int(broken[0]) if broken.size else 0
        return best_action


class CostGreedyPolicy(GreedyPolicy):
    name = "cost_greedy"

    def act(self, observation) -> int:
        state = np.asarray(observation[: len(self.env.dimensions)], dtype=np.int8)
        budget = int(observation[len(self.env.dimensions)])
        best_action = -1
        best_key = None
        for action in range(len(self.env.dimensions)):
            if int(state[action]) == 0 or self.env.action_costs[action] > budget:
                continue
            cost = self.env.action_costs[action]
            key = (self._score_drop(state, action) / cost, -cost, self.weights[action])
            if best_key is None or key > best_key:
                best_key = key
                best_action = action
        if best_action < 0:
            broken = np.flatnonzero(state == 1)
            return int(broken[0]) if broken.size else 0
        return best_action


class OptimalPolicy:
    name = "optimal"

    def __init__(self, env: RemediationEnv):
        self.env = env
        self.n_states = 1 << len(env.dimensions)
        self.budget_levels = env.budget + 1
        self.values = np.zeros((self.n_states, self.budget_levels), dtype=float)
        self.actions = np.full((self.n_states, self.budget_levels), -1, dtype=int)
        self._solve()

    def _state_from_bits(self, bits: int) -> np.ndarray:
        return np.array(
            [(bits >> index) & 1 for index in range(len(self.env.dimensions))], dtype=np.int8
        )

    def _bits(self, state) -> int:
        return self.env.bits_index(state)

    def _continuation(self, state, budget: int, score: float) -> float:
        if self.env.is_terminal(state, budget):
            return 0.0
        return float(self.values[self._bits(state), budget])

    def _expected_value(self, state, budget: int, action: int) -> float:
        cost = self.env.action_costs[action]
        next_budget = budget - int(cost)
        score = self.env.score_state(state)
        probability = self.env.success_probs[action]

        failure = -cost + self._continuation(state, next_budget, score)

        next_state = state.copy()
        next_state[action] = 0
        score_success = self.env.score_state(next_state)
        success_reward = score - score_success - cost
        if self.env.is_clean(next_state):
            success_reward += self.env.completion_bonus
        success = success_reward + self._continuation(next_state, next_budget, score_success)

        if action == self.env.unauth_index:
            side_state = next_state.copy()
            side_state[self.env.wwf_index] = 1
            score_side = self.env.score_state(side_state)
            side_reward = score - score_side - cost - self.env.side_effect_penalty
            if self.env.is_clean(side_state):
                side_reward += self.env.completion_bonus
            side = side_reward + self._continuation(side_state, next_budget, score_side)
            success = (
                (1 - self.env.side_effect_prob) * success
                + self.env.side_effect_prob * side
            )

        return probability * success + (1 - probability) * failure

    def _solve(self) -> None:
        for budget in range(1, self.budget_levels):
            for bits in range(self.n_states):
                state = self._state_from_bits(bits)
                if self.env.is_terminal(state, budget):
                    self.values[bits, budget] = 0.0
                    self.actions[bits, budget] = -1
                    continue
                best_value = -np.inf
                best_action = -1
                for action in range(len(self.env.dimensions)):
                    if int(state[action]) == 0 or self.env.action_costs[action] > budget:
                        continue
                    value = self._expected_value(state, budget, action)
                    if value > best_value:
                        best_value = value
                        best_action = action
                self.values[bits, budget] = best_value
                self.actions[bits, budget] = best_action

    def act(self, observation) -> int:
        bits = self.env.bits_index(observation)
        budget = int(observation[len(self.env.dimensions)])
        action = int(self.actions[bits, budget])
        if action < 0:
            broken = np.flatnonzero(np.asarray(observation[: len(self.env.dimensions)]) == 1)
            return int(broken[0]) if broken.size else 0
        return action

    def expected_value(self, observation) -> float:
        bits = self.env.bits_index(observation)
        budget = int(observation[len(self.env.dimensions)])
        return float(self.values[bits, budget])
