"""Turn a real posture scan into an ordered remediation plan using an RL or reference policy.

The policies act on the abstract 6-bit state the environment trains on. The planner maps a
VM's actual findings onto that state, asks the policy for one fix at a time, and reports
scores recomputed from the VM's own remaining findings, so the numbers match what
``POST /scan`` would return after each fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from riskvane.common.schema import Finding, PostureFacts
from riskvane.posture.scoring import cve_feed, scorer
from riskvane.posture.scoring.cve_feed import CveEntry
from riskvane.posture.training.synthetic import DIMENSIONS
from riskvane.rl.baselines import CostGreedyPolicy, GreedyPolicy, OptimalPolicy
from riskvane.rl.env import RemediationEnv

PolicyName = Literal["optimal", "greedy", "cost_greedy", "qlearning", "dqn"]
POLICY_NAMES: tuple[str, ...] = ("optimal", "greedy", "cost_greedy", "qlearning", "dqn")

DEFAULT_Q_TABLE_PATH = Path("var/rl/q_table.json")
DEFAULT_DQN_MODEL_PATH = Path("var/rl/dqn_model.zip")

ACTION_GUIDANCE = {
    "root_login": "Set `PermitRootLogin no` in sshd_config and reload sshd.",
    "password_auth": "Set `PasswordAuthentication no` in sshd_config (key-only login).",
    "cve_exposure": "Upgrade the vulnerable packages to their fixed versions.",
    "unauthenticated_service": (
        "Enable authentication on, or stop exposing, the unauthenticated services. "
        "Watch for world-writable files created during reconfiguration."
    ),
    "world_writable_files": "Remove the world-writable bit (`chmod o-w`) from flagged files.",
    "suid_binaries": "Drop the SUID bit (`chmod u-s`) from binaries that don't need it.",
}


class PolicyUnavailableError(RuntimeError):
    """Raised when a learned policy is requested but its trained artifact is missing."""


class RemediationStep(BaseModel):
    order: int
    dimension: str
    guidance: str
    finding_ids: list[str]
    cost: float
    success_probability: float
    score_before: int
    score_after: int


class RemediationPlan(BaseModel):
    vm_id: str
    policy: str
    budget: int
    initial_score: int
    expected_final_score: int
    budget_spent: float
    steps: list[RemediationStep] = Field(default_factory=list)
    unaddressed: list[str] = Field(
        default_factory=list,
        description="Risk dimensions still present after the plan (budget ran out).",
    )


def findings_by_dimension(findings: list[Finding]) -> dict[str, list[Finding]]:
    grouped: dict[str, list[Finding]] = {name: [] for name in DIMENSIONS}
    for finding in findings:
        if finding.category == "cve":
            grouped["cve_exposure"].append(finding)
        elif finding.id == "ssh-root-login":
            grouped["root_login"].append(finding)
        elif finding.id == "ssh-password-auth":
            grouped["password_auth"].append(finding)
        elif finding.id.startswith("exposed-unauthenticated-"):
            grouped["unauthenticated_service"].append(finding)
        elif finding.id.startswith("filesystem-world_writable-"):
            grouped["world_writable_files"].append(finding)
        elif finding.id.startswith("filesystem-suid-"):
            grouped["suid_binaries"].append(finding)
    return grouped


def state_from_findings(findings: list[Finding]) -> np.ndarray:
    grouped = findings_by_dimension(findings)
    return np.array([int(bool(grouped[name])) for name in DIMENSIONS], dtype=np.int8)


class RemediationPlanner:
    """Plans fixes for one VM at a time. Learned policies are loaded lazily on first use."""

    def __init__(
        self,
        q_table_path: Path | None = None,
        dqn_model_path: Path | None = None,
        feed: list[CveEntry] | None = None,
    ):
        self.env = RemediationEnv()
        self.feed = feed if feed is not None else cve_feed.load_feed()
        self.q_table_path = Path(q_table_path) if q_table_path else DEFAULT_Q_TABLE_PATH
        self.dqn_model_path = Path(dqn_model_path) if dqn_model_path else DEFAULT_DQN_MODEL_PATH
        self._policies: dict[str, object] = {}

    @property
    def max_budget(self) -> int:
        return self.env.budget

    def _policy(self, name: str):
        if name in self._policies:
            return self._policies[name]
        if name == "optimal":
            policy = OptimalPolicy(self.env)
        elif name == "greedy":
            policy = GreedyPolicy(self.env)
        elif name == "cost_greedy":
            policy = CostGreedyPolicy(self.env)
        elif name == "qlearning":
            if not self.q_table_path.exists():
                raise PolicyUnavailableError(
                    f"no trained Q-table at {self.q_table_path}; run riskvane-train-rl"
                )
            from riskvane.rl.qlearning import QLearningAgent

            agent = QLearningAgent.load(self.q_table_path)
            policy = _ActAdapter(agent.greedy_action)
        elif name == "dqn":
            if not self.dqn_model_path.exists():
                raise PolicyUnavailableError(
                    f"no trained DQN model at {self.dqn_model_path}; run riskvane-train-rl"
                )
            from riskvane.rl.dqn import DQNAgent

            agent = DQNAgent.load(self.dqn_model_path)
            policy = _ActAdapter(agent.act)
        else:
            raise ValueError(f"unknown policy: {name!r}")
        self._policies[name] = policy
        return policy

    def plan(
        self, facts: PostureFacts, policy: str = "optimal", budget: int | None = None
    ) -> RemediationPlan:
        budget = self.max_budget if budget is None else int(budget)
        if not 0 <= budget <= self.max_budget:
            raise ValueError(f"budget must be between 0 and {self.max_budget}")
        agent = self._policy(policy)
        starting_budget = budget

        findings = scorer.score(facts, self.feed).findings
        grouped = findings_by_dimension(findings)
        state = state_from_findings(findings)
        remaining = list(findings)
        initial_score = scorer.total_points(remaining)

        steps: list[RemediationStep] = []
        spent = 0.0
        while not self.env.is_terminal(state, budget):
            action = int(agent.act(np.array([*state, budget], dtype=np.int8)))
            valid = 0 <= action < len(DIMENSIONS) and state[action] == 1
            if not valid or self.env.action_costs[action] > budget:
                # A learned policy proposed an invalid move; stop rather than loop.
                break
            cost = self.env.action_costs[action]
            dimension = DIMENSIONS[action]
            cleared = {finding.id for finding in grouped[dimension]}
            score_before = scorer.total_points(remaining)
            remaining = [finding for finding in remaining if finding.id not in cleared]
            score_after = scorer.total_points(remaining)
            steps.append(
                RemediationStep(
                    order=len(steps) + 1,
                    dimension=dimension,
                    guidance=ACTION_GUIDANCE[dimension],
                    finding_ids=sorted(cleared),
                    cost=cost,
                    success_probability=self.env.success_probs[action],
                    score_before=score_before,
                    score_after=score_after,
                )
            )
            state[action] = 0
            budget -= int(cost)
            spent += cost

        return RemediationPlan(
            vm_id=facts.vm_id,
            policy=policy,
            budget=starting_budget,
            initial_score=initial_score,
            expected_final_score=scorer.total_points(remaining),
            budget_spent=spent,
            steps=steps,
            unaddressed=[name for i, name in enumerate(DIMENSIONS) if state[i]],
        )


class _ActAdapter:
    def __init__(self, act):
        self._act = act

    def act(self, observation) -> int:
        return self._act(observation)
