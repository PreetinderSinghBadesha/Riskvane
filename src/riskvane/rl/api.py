"""Remediation API: POST /remediate."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from riskvane.common.schema import PostureFacts
from riskvane.rl.planner import (
    POLICY_NAMES,
    PolicyName,
    PolicyUnavailableError,
    RemediationPlan,
    RemediationPlanner,
)


def create_router(
    q_table_path: Path | None = None,
    dqn_model_path: Path | None = None,
) -> APIRouter:
    router = APIRouter()
    planner = RemediationPlanner(q_table_path=q_table_path, dqn_model_path=dqn_model_path)

    @router.post("/remediate", response_model=RemediationPlan)
    def remediate(
        facts: PostureFacts,
        policy: Annotated[
            PolicyName, Query(description=f"One of {', '.join(POLICY_NAMES)}")
        ] = "optimal",
        budget: Annotated[int | None, Query(ge=0, le=planner.max_budget)] = None,
    ) -> RemediationPlan:
        try:
            return planner.plan(facts, policy=policy, budget=budget)
        except PolicyUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return router
