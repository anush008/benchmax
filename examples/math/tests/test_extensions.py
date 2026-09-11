from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmax.auth import StaticBearerAuth
from extensions.math_group_env import MathGroupEnv
from extensions.stress_test_env import (
    FAILURE_KEY,
    FAILURE_MODES,
    StressTestMathEnv,
)
from main import MathEnv

from benchmax.envs import BaseRollout, Example, RolloutFailure, RolloutRequest, canonical_example_id


def _write_rows(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            f"{json.dumps({'question': f'{index} + 1', 'answer': str(index + 1)})}\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def _rollout(
    rollout_id: str,
    *,
    failure: str | None = None,
) -> BaseRollout:
    example_args = {"answer": "42"}
    if failure is not None:
        example_args[FAILURE_KEY] = failure
    return BaseRollout(
        rollout_id=rollout_id,
        termination_reason="finished",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call", "type": "function"}],
            },
            {"role": "assistant", "content": "<answer>42</answer>"},
        ],
        example_args=example_args,
    )


def _example(failure: str) -> Example:
    payload = {
        "prompt_messages": [{"role": "user", "content": "6 * 7"}],
        "answer": "42",
        FAILURE_KEY: failure,
    }
    return Example(id=canonical_example_id(payload), payload=payload)


@pytest.mark.asyncio
async def test_group_extension_scores_only_at_group_boundary() -> None:
    env = MathGroupEnv()
    rollouts = [_rollout("a"), _rollout("b")]

    assert await env.compute_reward(rollouts[0]) is None
    assert await env.compute_group_rewards(rollouts) == {
        "a": {"correctness": 1.0},
        "b": {"correctness": 1.0},
    }


@pytest.mark.asyncio
async def test_stress_dataset_keeps_first_example_healthy_then_cycles_failures(
    tmp_path: Path,
) -> None:
    _write_rows(tmp_path / "train.jsonl", len(FAILURE_MODES) + 1)
    env = StressTestMathEnv()

    dataset = await env.create_dataset("train", tmp_path)

    assert FAILURE_KEY not in dataset[0].payload
    assert tuple(example.payload[FAILURE_KEY] for example in tuple(dataset)[1:]) == FAILURE_MODES


@pytest.mark.asyncio
async def test_stress_scenarios_isolate_the_requested_failure(tmp_path: Path) -> None:
    _write_rows(tmp_path / "train.jsonl", 6)

    partial = await StressTestMathEnv(scenario="partial_sibling").create_dataset("train", tmp_path)
    all_empty = await StressTestMathEnv(scenario="all_siblings_empty").create_dataset(
        "train",
        tmp_path,
    )

    assert [example.payload.get(FAILURE_KEY) for example in partial] == [
        None,
        "partial_sibling",
        None,
        None,
        None,
        None,
    ]
    assert [example.payload.get(FAILURE_KEY) for example in all_empty] == [
        None,
        "init_rollout",
        "init_rollout",
        "init_rollout",
        None,
        None,
    ]


@pytest.mark.asyncio
async def test_stress_context_and_reward_failures_are_labeled() -> None:
    env = StressTestMathEnv()

    with pytest.raises(RolloutFailure, match="rollout setup failed") as setup:
        async with env.rollout_context("setup", _example("init_rollout")):
            pass
    assert setup.value.termination_reason == "harness_error"

    with pytest.raises(RolloutFailure, match="rollout cleanup failed") as cleanup:
        async with env.rollout_context("cleanup", _example("release_rollout")):
            pass
    assert cleanup.value.termination_reason == "harness_error"

    with pytest.raises(RuntimeError, match="tool execution failed"):
        async with env.rollout_context("tool", _example("run_tool")):
            await env.run_tool("tool", "multiply", a=6, b=7)

    with pytest.raises(RolloutFailure, match="reward service failed") as reward:
        await env.compute_reward(_rollout("reward", failure="compute_reward"))
    assert reward.value.termination_reason == "judge_error"

    with pytest.raises(RolloutFailure, match="group reward service failed") as group:
        await env.compute_group_rewards([_rollout("group", failure="compute_group_reward")])
    assert group.value.termination_reason == "judge_error"


@pytest.mark.asyncio
async def test_stress_crash_happens_once_then_recovers(monkeypatch) -> None:
    env = StressTestMathEnv()
    example = _example("crash_once")
    request = SimpleNamespace(example=example, rollout_id="rollout")

    async def successful_rollout(self, received):
        assert received is request
        return _rollout(received.rollout_id)

    monkeypatch.setattr(MathEnv, "run_rollout", successful_rollout)

    with pytest.raises(RuntimeError, match="crash once"):
        await env.run_rollout(request)

    recovered = await env.run_rollout(request)
    assert recovered.termination_reason == "finished"


@pytest.mark.asyncio
async def test_stress_partial_sibling_settles_only_first_member(monkeypatch) -> None:
    env = StressTestMathEnv()
    example = _example("partial_sibling")
    requests = [
        RolloutRequest(
            rollout_id=rollout_id,
            example=example,
            model="test-model",
            base_url=f"http://model.test/sessions/{rollout_id}/v1",
            model_auth=StaticBearerAuth(f"key-{rollout_id}"),
        )
        for rollout_id in ("failed-sibling", "successful-sibling")
    ]

    async def successful_rollout(self, request):
        return replace(
            _rollout(request.rollout_id, failure="partial_sibling"),
            rewards={"correctness": 1.0},
        )

    monkeypatch.setattr(MathEnv, "run_rollout", successful_rollout)

    outcomes = await env.run_group(requests)

    assert outcomes["failed-sibling"].termination_reason == "harness_error"
    assert outcomes["failed-sibling"].rewards == {}
    assert outcomes["failed-sibling"].error == (
        "stress test: partial sibling failed before model execution"
    )
    assert outcomes["successful-sibling"].termination_reason == "finished"
    assert outcomes["successful-sibling"].rewards == {"correctness": 1.0}
    assert outcomes["successful-sibling"].error is None


@pytest.mark.asyncio
async def test_stress_all_siblings_empty_settles_every_member() -> None:
    env = StressTestMathEnv(scenario="all_siblings_empty")
    example = _example("init_rollout")
    requests = [
        RolloutRequest(
            rollout_id=rollout_id,
            example=example,
            model="test-model",
            base_url=f"http://model.test/sessions/{rollout_id}/v1",
            model_auth=StaticBearerAuth(f"key-{rollout_id}"),
        )
        for rollout_id in ("empty-a", "empty-b")
    ]

    outcomes = await env.run_group(requests)

    assert set(outcomes) == {"empty-a", "empty-b"}
    assert all(outcome.termination_reason == "harness_error" for outcome in outcomes.values())
    assert all(outcome.rewards == {} for outcome in outcomes.values())
    assert all(
        outcome.error == "stress test: rollout setup failed" for outcome in outcomes.values()
    )
