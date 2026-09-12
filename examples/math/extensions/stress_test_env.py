"""Inject representative recoverable and terminal failures into MathEnv."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from main import MathEnv, run_cli

from benchmax.envs import (
    BaseRollout,
    Dataset,
    Example,
    JsonRow,
    RolloutFailure,
    RolloutOutcome,
    RolloutRequest,
    canonical_example_id,
)

FAILURE_KEY = "_stress_failure"
SCENARIOS = ("cycle", "partial_sibling", "all_siblings_empty")
FAILURE_MODES = (
    "partial_sibling",
    "crash_once",
    "init_rollout",
    "run_tool",
    "compute_reward",
    "release_rollout",
    "compute_group_reward",
)


class StressTestMathEnv(MathEnv):
    """Inject a selected failure pattern while keeping item zero healthy."""

    def __init__(self, *, scenario: str = "cycle") -> None:
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown stress scenario: {scenario}")
        self._scenario = scenario
        self._active_failures: dict[str, str] = {}
        self._crashed_examples: set[str] = set()
        self._partial_sibling_failures: set[str] = set()

    async def create_dataset(
        self,
        split,
        base_dir,
        *,
        max_examples: int | None = None,
    ) -> Dataset[JsonRow]:
        dataset = await super().create_dataset(
            split,
            base_dir,
            max_examples=max_examples,
        )
        examples: list[Example[JsonRow]] = []
        for index, example in enumerate(dataset):
            payload = dict(example.payload)
            failure = self._failure_for_index(index)
            if failure is not None:
                payload[FAILURE_KEY] = failure
            examples.append(
                Example(
                    id=canonical_example_id(payload),
                    payload=payload,
                )
            )
        return Dataset(examples)

    def _failure_for_index(self, index: int) -> str | None:
        if index == 0:
            return None  # Validation always selects one healthy example.
        if self._scenario == "partial_sibling":
            return "partial_sibling" if index == 1 else None
        if self._scenario == "all_siblings_empty":
            return "init_rollout" if index <= 3 else None
        return FAILURE_MODES[(index - 1) % len(FAILURE_MODES)]

    async def run_group(
        self,
        requests: Sequence[RolloutRequest[JsonRow]],
    ) -> Mapping[str, RolloutOutcome]:
        """Fail only the first sibling for a partial-sibling stress example."""

        victim = None
        if requests and requests[0].example.payload.get(FAILURE_KEY) == "partial_sibling":
            victim = requests[0].rollout_id
            self._partial_sibling_failures.add(victim)
        try:
            return await super().run_group(requests)
        finally:
            if victim is not None:
                self._partial_sibling_failures.discard(victim)

    async def run_rollout(
        self,
        request: RolloutRequest[JsonRow],
    ) -> BaseRollout:
        failure = request.example.payload.get(FAILURE_KEY)
        if failure == "partial_sibling" and request.rollout_id in self._partial_sibling_failures:
            raise RolloutFailure(
                "harness_error",
                "stress test: partial sibling failed before model execution",
            )
        if failure == "crash_once" and request.example.id not in self._crashed_examples:
            self._crashed_examples.add(request.example.id)
            raise RuntimeError("stress test: crash once before model execution")
        return await super().run_rollout(request)

    def rollout_context(
        self,
        rollout_id: str,
        example: Example[JsonRow],
    ) -> _StressRolloutContext:
        return _StressRolloutContext(self, rollout_id, example)

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> str:
        if self._active_failures.get(rollout_id) == "run_tool":
            raise RuntimeError("stress test: tool execution failed")
        return await super().run_tool(rollout_id, tool_name, **tool_args)

    async def compute_reward(self, rollout: BaseRollout) -> dict[str, float]:
        if rollout.example_args.get(FAILURE_KEY) == "compute_reward":
            raise RolloutFailure("judge_error", "stress test: reward service failed")
        return await super().compute_reward(rollout)

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> None:
        if any(
            rollout.example_args.get(FAILURE_KEY) == "compute_group_reward" for rollout in rollouts
        ):
            raise RolloutFailure(
                "judge_error",
                "stress test: group reward service failed",
            )
        return None


class _StressRolloutContext:
    def __init__(
        self,
        env: StressTestMathEnv,
        rollout_id: str,
        example: Example[JsonRow],
    ) -> None:
        self._env = env
        self._rollout_id = rollout_id
        self._failure = str(example.payload.get(FAILURE_KEY, ""))

    async def __aenter__(self) -> None:
        if self._failure == "init_rollout":
            raise RolloutFailure(
                "harness_error",
                "stress test: rollout setup failed",
            )
        self._env._active_failures[self._rollout_id] = self._failure

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        failure = self._env._active_failures.pop(self._rollout_id, "")
        if exc is None and failure == "release_rollout":
            raise RolloutFailure(
                "harness_error",
                "stress test: rollout cleanup failed",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", choices=SCENARIOS, default="cycle")
    scenario_args, remaining = parser.parse_known_args(argv)
    return run_cli(
        StressTestMathEnv,
        run_name=f"math-stress-{scenario_args.scenario.replace('_', '-')}",
        constructor_args={"scenario": scenario_args.scenario},
        argv=remaining,
    )


if __name__ == "__main__":
    sys.exit(main())
