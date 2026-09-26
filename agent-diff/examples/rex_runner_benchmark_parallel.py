"""Run Agent-Diff REx tasks concurrently with per-task checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_diff import AgentDiff
from agents.rex_runner import RExConfig, OpenAICompatibleModelClient
from rex_runner_benchmark import (
    AGENT_NAME,
    TRACE_SCHEMA_VERSION,
    RExCheckpointStore,
    RExTaskCheckpoint,
    build_parser,
    print_summary,
    resolve_checkpoint_dir,
    resolve_compression_log_dir,
    run_example,
    select_examples,
    utc_now,
)

from datasets import load_dataset


def build_parallel_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("REX_WORKERS", "8")),
        help="Maximum number of tasks to execute concurrently (default: 8).",
    )
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


def build_agent_config(args: argparse.Namespace) -> RExConfig:
    return RExConfig(
        max_depth=args.max_depth,
        max_steps_per_frame=args.max_steps_per_frame,
        max_turns_per_step=args.max_turns_per_step,
        max_tool_calls=args.max_tool_calls,
        compression_mode=args.compression_mode,
        compression_frame_trigger_tokens=args.compression_frame_trigger_tokens,
    )


def execute_task(
    *,
    example: dict[str, Any],
    checkpoint: RExTaskCheckpoint,
    key: str,
    args: argparse.Namespace,
    agent_config: RExConfig,
    compression_log_root: Path,
) -> dict[str, Any]:
    # Keep clients worker-local so no mutable HTTP state is shared across tasks.
    client = AgentDiff(
        api_key=args.agent_diff_api_key or None,
        base_url=args.agent_diff_base_url,
    )
    model_client = OpenAICompatibleModelClient(
        model=args.model,
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
    )
    return run_example(
        example=example,
        client=client,
        model_client=model_client,
        agent_config=agent_config,
        checkpoint=checkpoint,
        key=key,
        model_name=args.model,
        compression_log_root=compression_log_root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parallel_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    agent_config = build_agent_config(args)

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    selected = select_examples(dataset, args)
    if not selected:
        print("No benchmark tasks matched the requested filters.", file=sys.stderr)
        return 2

    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "agent": AGENT_NAME,
        "model": args.model,
        "openai_base_url": args.openai_base_url.rstrip("/"),
        "agent_diff_base_url": args.agent_diff_base_url.rstrip("/"),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "config": asdict(agent_config),
    }
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_store = RExCheckpointStore(
        resolve_checkpoint_dir(args, repo_root), metadata
    )
    compression_log_root = resolve_compression_log_dir(
        args, repo_root, checkpoint_store.directory
    )

    scheduled: list[tuple[dict[str, Any], str, RExTaskCheckpoint]] = []
    task_checkpoints: list[RExTaskCheckpoint] = []
    for example in selected:
        test_id = str(example["test_id"])
        key = checkpoint_store.task_key(args.model, test_id)
        checkpoint = checkpoint_store.for_task(test_id=test_id, task_key=key)
        task_checkpoints.append(checkpoint)
        if checkpoint.is_completed():
            print(f"[SKIP] {example.get('test_name', test_id)} already completed")
            continue
        scheduled.append((example, key, checkpoint))

    worker_count = min(args.workers, len(scheduled)) if scheduled else 0
    if worker_count:
        print(
            f"Executing {len(scheduled)} tasks with {worker_count} workers; "
            f"checkpoints: {checkpoint_store.directory}"
        )
        futures: dict[Future[dict[str, Any]], str] = {}
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="agent-diff-rex",
        ) as executor:
            for example, key, checkpoint in scheduled:
                future = executor.submit(
                    execute_task,
                    example=example,
                    checkpoint=checkpoint,
                    key=key,
                    args=args,
                    agent_config=agent_config,
                    compression_log_root=compression_log_root,
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
        for checkpoint in task_checkpoints
        if (result := checkpoint.result()) is not None
    ]
    hard_results = [
        result for result in results if int(result.get("task_horizon", 0) or 0) >= 5
    ]
    print()
    print_summary("All selected tasks", results)
    print_summary("Hard tasks (task_horizon >= 5)", hard_results)
    print(f"Checkpoint directory: {checkpoint_store.directory}")
    return 0 if len(results) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
