import numpy as np
import pytest
from fastapi.testclient import TestClient

from riskvane.api.app import create_app
from riskvane.posture.scoring import scorer
from riskvane.posture.training.synthetic import DIMENSIONS, build_facts, generate_profiles
from riskvane.rl.env import RemediationEnv
from riskvane.rl.planner import PolicyUnavailableError, RemediationPlanner, state_from_findings
from riskvane.rl.qlearning import QLearningAgent


def _all_broken_facts():
    return build_facts("planner-vm", {name: True for name in DIMENSIONS})


@pytest.fixture(scope="module")
def planner(tmp_path_factory) -> RemediationPlanner:
    missing = tmp_path_factory.mktemp("rl") / "missing"
    return RemediationPlanner(q_table_path=missing / "q.json", dqn_model_path=missing / "d.zip")


def test_findings_map_back_to_the_profile_for_every_synthetic_vm():
    for index, profile in enumerate(generate_profiles()):
        facts = build_facts(f"vm-{index}", profile)
        state = state_from_findings(scorer.score(facts).findings)
        assert state.tolist() == [int(profile[name]) for name in DIMENSIONS]


def test_clean_vm_gets_an_empty_plan(planner):
    facts = build_facts("clean", {name: False for name in DIMENSIONS})

    plan = planner.plan(facts)

    assert plan.steps == []
    assert plan.initial_score == 0
    assert plan.expected_final_score == 0
    assert plan.unaddressed == []


@pytest.mark.parametrize("policy", ["optimal", "greedy", "cost_greedy"])
def test_full_budget_plan_clears_every_dimension(planner, policy):
    plan = planner.plan(_all_broken_facts(), policy=policy)

    assert plan.expected_final_score == 0
    assert plan.unaddressed == []
    assert sorted(step.dimension for step in plan.steps) == sorted(DIMENSIONS)
    assert plan.budget_spent == sum(step.cost for step in plan.steps) <= plan.budget
    for previous, step in zip(plan.steps, plan.steps[1:]):
        assert step.score_before == previous.score_after
    assert all(step.finding_ids for step in plan.steps)


def test_tight_budget_leaves_dimensions_unaddressed(planner):
    plan = planner.plan(_all_broken_facts(), budget=3)

    assert plan.budget_spent <= 3
    assert plan.unaddressed
    assert len(plan.steps) + len(plan.unaddressed) == len(DIMENSIONS)


def test_zero_budget_plans_nothing(planner):
    plan = planner.plan(_all_broken_facts(), budget=0)

    assert plan.steps == []
    assert sorted(plan.unaddressed) == sorted(DIMENSIONS)


def test_out_of_range_budget_is_rejected(planner):
    with pytest.raises(ValueError):
        planner.plan(_all_broken_facts(), budget=planner.max_budget + 1)


def test_learned_policy_without_artifact_is_unavailable(planner):
    with pytest.raises(PolicyUnavailableError):
        planner.plan(_all_broken_facts(), policy="qlearning")


def test_trained_q_table_produces_a_valid_plan(tmp_path):
    env = RemediationEnv()
    agent = QLearningAgent(n_budgets=env.budget + 1, costs=env.action_costs, seed=0)
    agent.train(env, episodes=300)
    q_path = tmp_path / "q_table.json"
    agent.save(q_path)

    plan = RemediationPlanner(q_table_path=q_path).plan(_all_broken_facts(), policy="qlearning")

    assert plan.policy == "qlearning"
    assert plan.steps
    assert len({step.dimension for step in plan.steps}) == len(plan.steps)
    assert plan.budget_spent <= plan.budget


def test_completion_is_defined_by_state_not_score(monkeypatch):
    env = RemediationEnv()
    clean_score = 7.0  # what the trained XGBoost model predicts for a hardened VM
    monkeypatch.setattr(
        env, "score_state", lambda state: clean_score + 20.0 * int(np.asarray(state)[0])
    )
    env.reset(seed=0, options={"profile": {name: name == "root_login" for name in DIMENSIONS}})

    _, reward, terminated, _, info = env.step(0)

    assert info["score_after"] == clean_score
    assert terminated is True
    assert reward == pytest.approx(20 - 1 + env.completion_bonus)


def _payload(facts) -> dict:
    return facts.model_dump(mode="json")


def test_remediate_endpoint_returns_a_plan(tmp_path):
    client = TestClient(create_app(training_log_path=tmp_path / "training.jsonl"))

    response = client.post("/remediate?budget=8", json=_payload(_all_broken_facts()))

    assert response.status_code == 200
    body = response.json()
    assert body["policy"] == "optimal"
    assert body["vm_id"] == "planner-vm"
    assert body["expected_final_score"] == 0
    assert len(body["steps"]) == len(DIMENSIONS)


def test_remediate_endpoint_reports_missing_learned_policy(tmp_path):
    client = TestClient(
        create_app(
            training_log_path=tmp_path / "training.jsonl",
            q_table_path=tmp_path / "missing.json",
        )
    )

    response = client.post("/remediate?policy=qlearning", json=_payload(_all_broken_facts()))

    assert response.status_code == 503


def test_remediate_endpoint_validates_query(tmp_path):
    client = TestClient(create_app(training_log_path=tmp_path / "training.jsonl"))

    assert client.post("/remediate?budget=99", json=_payload(_all_broken_facts())).status_code == 422
    assert client.post("/remediate?policy=nope", json=_payload(_all_broken_facts())).status_code == 422
