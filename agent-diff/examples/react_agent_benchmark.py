"""Run ReAct tasks concurrently with one resumable checkpoint per task."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_diff import AgentDiff, BashExecutorProxy
from agent_benchmark_common import (
    CheckpointStore,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    TRACE_SCHEMA_VERSION,
    TaskCheckpoint,
    print_summary,
    resolve_checkpoint_dir,
    run_agent_example,
    select_examples,
    utc_now,
)
from agents.rex_runner import OpenAICompatibleModelClient
from agents.react import ReactConfig, run_react_agent
from datasets import load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument(
        "--agent-diff-base-url",
        default=os.getenv("AGENT_DIFF_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--agent-diff-api-key", default=os.getenv("AGENT_DIFF_API_KEY", "")
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--test-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-task-horizon", type=int, default=0)
    parser.add_argument("--checkpoint-dir", "--checkpoint-path", dest="checkpoint_dir")
    parser.add_argument(
        "--trace-dir",
        default="examples/evaluation_outputs/react_agent/checkpoints",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("REACT_WORKERS", "8")),
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.model:
        parser.error("--model or OPENAI_MODEL is required")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.min_task_horizon < 0:
        parser.error("--min-task-horizon cannot be negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")


def execute_task(
    *,
    example: dict[str, Any],
    checkpoint: TaskCheckpoint,
    key: str,
    args: argparse.Namespace,
    agent_config: ReactConfig,
) -> dict[str, Any]:
    client = AgentDiff(
        api_key=args.agent_diff_api_key or None,
        base_url=args.agent_diff_base_url,
    )
    model_client = OpenAICompatibleModelClient(
        model=args.model,
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
    )
    return run_agent_example(
        example=example,
        client=client,
        model_client=model_client,
        agent_config=agent_config,
        checkpoint=checkpoint,
        key=key,
        model_name=args.model,
        run_agent=run_react_agent,
        executor_factory=BashExecutorProxy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    agent_config = ReactConfig(max_iterations=args.max_iterations)

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    selected = select_examples(dataset, args)
    if not selected:
        print("No benchmark tasks matched the requested filters.", file=sys.stderr)
        return 2

    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "agent": "react",
        "model": args.model,
        "openai_base_url": args.openai_base_url.rstrip("/"),
        "agent_diff_base_url": args.agent_diff_base_url.rstrip("/"),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "config": asdict(agent_config),
    }
    repo_root = Path(__file__).resolve().parents[1]
    store = CheckpointStore(resolve_checkpoint_dir(args, repo_root), metadata)

    scheduled: list[tuple[dict[str, Any], str, TaskCheckpoint]] = []
    checkpoints: list[TaskCheckpoint] = []
    for example in selected:
        test_id = str(example["test_id"])
        key = store.task_key(args.model, test_id)
        checkpoint = store.for_task(test_id=test_id, task_key=key)
        checkpoints.append(checkpoint)
        if checkpoint.is_completed():
            print(f"[SKIP] {example.get('test_name', test_id)} already completed")
            continue
        scheduled.append((example, key, checkpoint))

    worker_count = min(args.workers, len(scheduled)) if scheduled else 0
    if worker_count:
        print(
            f"Executing {len(scheduled)} ReAct tasks with {worker_count} workers; "
            f"checkpoints: {store.directory}"
        )
        futures: dict[Future[dict[str, Any]], str] = {}
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="agent-diff-react",
        ) as executor:
            for example, key, checkpoint in scheduled:
                future = executor.submit(
                    execute_task,
                    example=example,
                    checkpoint=checkpoint,
                    key=key,
                    args=args,
                    agent_config=agent_config,
                )
                futures[future] = str(example["test_id"])
            for future in as_completed(futures):
                test_id = futures[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - isolate worker failures
                    print(
                        f"[WORKER ERROR] {test_id}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

    results = [
        result
        for checkpoint in checkpoints
        if (result := checkpoint.result()) is not None
    ]
    hard_results = [
        result for result in results if int(result.get("task_horizon", 0) or 0) >= 5
    ]
    print()
    print_summary("All selected tasks", results)
    print_summary("Hard tasks (task_horizon >= 5)", hard_results)
    print(f"Checkpoint directory: {store.directory}")
    return 0 if len(results) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
