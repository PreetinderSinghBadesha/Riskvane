import numpy as np
import pytest

from riskvane.posture.training.synthetic import DIMENSIONS, generate_profiles
from riskvane.rl.baselines import GreedyPolicy, OptimalPolicy
from riskvane.rl.env import RemediationEnv


def _profile(**overrides) -> dict[str, bool]:
    profile = {name: False for name in DIMENSIONS}
    profile.update(overrides)
    return profile


def _policy_values(env: RemediationEnv, actions) -> np.ndarray:
    n_states = 1 << len(DIMENSIONS)
    levels = env.budget + 1
    values = np.zeros((n_states, levels), dtype=float)

    for budget in range(1, levels):
        for bits in range(n_states):
            state = np.array(
                [(bits >> index) & 1 for index in range(len(DIMENSIONS))], dtype=np.int8
            )
            score = env.score_state(state)
            if score == 0 or not env.has_affordable_broken_action(state, budget):
                continue
            action = int(actions[bits, budget])
            if action < 0:
                continue

            cost = env.action_costs[action]
            next_budget = budget - int(cost)
            probability = env.success_probs[action]

            def continuation(next_state, remaining_budget, next_score):
                if (
                    next_score == 0
                    or remaining_budget == 0
                    or not env.has_affordable_broken_action(next_state, remaining_budget)
                ):
                    return 0.0
                return values[env.bits_index(next_state), remaining_budget]

            failure = -cost + continuation(state, next_budget, score)

            next_state = state.copy()
            next_state[action] = 0
            success_score = env.score_state(next_state)
            success_reward = score - success_score - cost
            if success_score == 0:
                success_reward += env.completion_bonus
            success = success_reward + continuation(next_state, next_budget, success_score)

            if action == env.unauth_index:
                side_state = next_state.copy()
                side_state[env.wwf_index] = 1
                side_score = env.score_state(side_state)
                side_reward = score - side_score - cost - env.side_effect_penalty
                if side_score == 0:
                    side_reward += env.completion_bonus
                side = side_reward + continuation(side_state, next_budget, side_score)
                success = (
                    (1 - env.side_effect_prob) * success + env.side_effect_prob * side
                )

            values[bits, budget] = probability * success + (1 - probability) * failure

    return values


def test_value_iteration_matches_analytic_single_action_value():
    env = RemediationEnv()
    policy = OptimalPolicy(env)
    observation = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.int8)

    expected = 0.95 * (20 - 0 - 1 + 10) + 0.05 * (-1)

    assert policy.expected_value(observation) == pytest.approx(expected)
    assert policy.act(observation) == 0


def test_severity_greedy_is_strictly_worse_than_optimal():
    env = RemediationEnv()
    optimal = OptimalPolicy(env)
    greedy = GreedyPolicy(env)

    greedy_actions = np.full(optimal.actions.shape, -1, dtype=int)
    for budget in range(1, env.budget + 1):
        for bits in range(1 << len(DIMENSIONS)):
            state = np.array(
                [(bits >> index) & 1 for index in range(len(DIMENSIONS))], dtype=np.int8
            )
            score = env.score_state(state)
            if score == 0 or not env.has_affordable_broken_action(state, budget):
                continue
            observation = np.array([*state, budget], dtype=np.int8)
            greedy_actions[bits, budget] = greedy.act(observation)

    greedy_values = _policy_values(env, greedy_actions)
    profiles = [profile for profile in generate_profiles() if any(profile.values())]

    optimal_mean = np.mean(
        [
            optimal.values[env.bits_index(env.profile_to_state(profile)), env.budget]
            for profile in profiles
        ]
    )
    greedy_mean = np.mean(
        [
            greedy_values[env.bits_index(env.profile_to_state(profile)), env.budget]
            for profile in profiles
        ]
    )

    assert greedy_mean < optimal_mean
