from contextlib import nullcontext
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai.model import Model

from inspect_evals.gaia.rex_runner import (
    RExConfig,
    RExPlanPatch,
    RExRunner,
    RExStep,
    RExStepResult,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("successful_batch", [False, True])
async def test_only_successful_batch_restores_recovery(
    monkeypatch: pytest.MonkeyPatch, successful_batch: bool
) -> None:
    runner = RExRunner(
        model=cast(Model, Mock()),
        messages=[],
        tools=[],
        config=RExConfig(
            max_steps_per_frame=8,
            max_depth=3,
            max_turns_per_step=1,
            max_planning_turns=1,
            max_tool_calls=100,
            max_observation_chars=100,
            planning_tools=False,
        ),
    )
    monkeypatch.setattr(
        "inspect_evals.gaia.rex_runner.span", lambda *args, **kwargs: nullcontext()
    )

    def batch(*goals: str, complete: bool = True) -> RExPlanPatch:
        return RExPlanPatch(
            [RExStep(goal, goal, "execute") for goal in goals], complete=complete
        )

    plans = [batch("Initial attempt", "Stale suffix")]
    if successful_batch:
        plans.extend(
            [
                batch("Recovery progress", complete=False),
                batch("Later failure", "Unused suffix"),
                batch("Later recovery"),
            ]
        )
    else:
        plans.append(batch("Recovery progress", "Later failure", "Unused suffix"))
    failure = RExStepResult("failed", "Source unavailable")
    outcomes = [failure, RExStepResult("success"), failure]
    if successful_batch:
        outcomes.append(RExStepResult("success"))
    plan = AsyncMock(side_effect=plans)
    execute = AsyncMock(side_effect=outcomes)
    event = Mock()
    monkeypatch.setattr(runner, "_plan_batch", plan)
    monkeypatch.setattr(runner, "_run_step", execute)
    monkeypatch.setattr(runner, "_event", event)

    result = await runner._run_frame_body("root", "Goal", depth=1)

    assert result.status == ("success" if successful_batch else "failed")
    assert plan.await_count == len(plans)
    assert [call.kwargs["step"].task for call in execute.await_args_list] == (
        ["Initial attempt", "Recovery progress", "Later failure"]
        + (["Later recovery"] if successful_batch else [])
    )
    assert [call.kwargs["recovery"] is not None for call in plan.await_args_list] == (
        [False, True, False, True] if successful_batch else [False, True]
    )
    if successful_batch:
        assert plan.await_args.kwargs["recovery"].failed_step_goal == "Later failure"
    aborts = [call.kwargs["artifact"] for call in event.call_args_list]
    assert [abort["recovery_exhausted"] for abort in aborts] == [
        False,
        not successful_batch,
    ]
    assert aborts[-1]["dropped_steps"][0]["step_goal"] == "Unused suffix"
