import numpy as np
import pytest

from riskvane.posture.training.synthetic import DIMENSIONS
from riskvane.rl.env import RemediationEnv


def _profile(**overrides) -> dict[str, bool]:
    profile = {name: False for name in DIMENSIONS}
    profile.update(overrides)
    return profile


def test_reset_returns_valid_non_clean_observation_with_budget():
    env = RemediationEnv()

    observation, info = env.reset(seed=0)

    assert env.observation_space.contains(observation)
    assert observation.shape == (len(DIMENSIONS) + 1,)
    assert info["score"] > 0
    assert info["budget"] == 8
    assert observation[-1] == 8
    assert any(int(value) for value in observation[: len(DIMENSIONS)])


def test_successful_reward_is_score_drop_minus_cost_plus_bonus():
    env = RemediationEnv()
    env.reset(seed=0, options={"profile": _profile(root_login=True)})

    observation, reward, terminated, truncated, info = env.step(0)

    assert info["score_before"] == 20
    assert info["score_after"] == 0
    assert info["success"] is True
    assert info["cost"] == 1
    assert reward == pytest.approx(20 - 0 - 1 + 10)
    assert terminated is True
    assert truncated is False
    assert observation[-1] == 7


def test_non_terminal_reward_uses_score_difference():
    env = RemediationEnv()
    env.reset(seed=0, options={"profile": _profile(root_login=True, password_auth=True)})

    _, first_reward, first_terminated, _, first_info = env.step(0)
    _, second_reward, second_terminated, _, second_info = env.step(1)

    assert first_info["score_before"] == 30
    assert first_info["score_after"] == 10
    assert first_reward == pytest.approx(30 - 10 - 1)
    assert first_terminated is False
    assert second_info["score_after"] == 0
    assert second_reward == pytest.approx(10 - 0 - 1 + 10)
    assert second_terminated is True


def test_fixed_action_is_penalized_noop_without_cost():
    env = RemediationEnv()
    observation, _ = env.reset(seed=0, options={"profile": _profile(root_login=True)})

    next_observation, reward, terminated, truncated, info = env.step(1)

    assert info["noop"] is True
    assert info["cost"] == 0
    assert info["score_before"] == info["score_after"] == 20
    assert reward == pytest.approx(-2)
    assert terminated is False
    assert truncated is False
    assert next_observation[-1] == observation[-1]
    assert (next_observation[: len(DIMENSIONS)] == observation[: len(DIMENSIONS)]).all()


def test_stochastic_failure_leaves_state_and_charges_cost():
    env = RemediationEnv()
    env.reset(seed=0, options={"profile": _profile(cve_exposure=True)})

    observation, reward, terminated, truncated, info = env.step(2)

    assert info["success"] is False
    assert info["score_before"] == info["score_after"] == 100
    assert info["cost"] == 2
    assert reward == pytest.approx(-2)
    assert terminated is False
    assert truncated is False
    assert observation[-1] == 6
    assert (observation[: len(DIMENSIONS)] == np.array([0, 0, 1, 0, 0, 0])).all()


def test_budget_exhaustion_terminates_without_bonus():
    env = RemediationEnv()
    env.reset(seed=0, options={"profile": _profile(cve_exposure=True), "budget": 2})

    observation, reward, terminated, truncated, info = env.step(2)

    assert terminated is True
    assert truncated is False
    assert info["score_after"] == 100
    assert reward == pytest.approx(-2)
    assert observation[-1] == 0


def test_side_effect_reverts_world_writable_files():
    env = RemediationEnv()
    env.reset(seed=3, options={"profile": _profile(unauthenticated_service=True)})

    observation, reward, terminated, _truncated, info = env.step(3)

    assert info["success"] is True
    assert info["side_effect"] is True
    assert info["score_before"] == 80
    assert info["score_after"] == 50
    assert reward == pytest.approx(80 - 50 - 2 - 8)
    assert observation[4] == 1
    assert terminated is False


def test_success_without_side_effect_completes_episode():
    env = RemediationEnv()
    env.reset(seed=11, options={"profile": _profile(unauthenticated_service=True)})

    observation, reward, terminated, _truncated, info = env.step(3)

    assert info["success"] is True
    assert info["side_effect"] is False
    assert info["score_after"] == 0
    assert reward == pytest.approx(80 - 0 - 2 + 10)
    assert observation[4] == 0
    assert terminated is True


def test_truncates_after_max_steps():
    env = RemediationEnv()
    env.reset(seed=0, options={"profile": _profile(root_login=True)})

    truncated = False
    for _ in range(12):
        _, reward, terminated, truncated, _ = env.step(1)
        assert reward == pytest.approx(-2)
        assert terminated is False

    assert truncated is True


def test_reset_never_selects_all_clean_profile():
    env = RemediationEnv()

    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        assert any(int(value) for value in observation[: len(DIMENSIONS)])


def test_budget_is_part_of_observation():
    env = RemediationEnv()
    env.reset(seed=0, options={"profile": _profile(root_login=True), "budget": 5})

    observation, _ = env.reset(
        seed=0, options={"profile": _profile(root_login=True), "budget": 5}
    )

    assert observation[-1] == 5
