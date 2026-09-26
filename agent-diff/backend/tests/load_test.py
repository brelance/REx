#!/usr/bin/env python3
"""
Load test for the diff-the-universe platform.

Measures:
- Environment creation time
- Message sending time
- Diff evaluation time
- Throughput under concurrent load

Usage:
    python tests/load_test.py --base-url http://localhost:8000 --concurrency 5 --requests 20
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class TestResult:
    """Result of a single test run."""

    env_id: str
    run_id: str
    init_time: float
    start_run_time: float
    action_time: float
    evaluate_time: float
    total_time: float
    success: bool
    error: str | None = None


@dataclass
class LoadTestResults:
    """Aggregated results from load test."""

    results: list[TestResult] = field(default_factory=list)

    @property
    def successful(self) -> list[TestResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> list[TestResult]:
        return [r for r in self.results if not r.success]

    def summary(self) -> dict[str, Any]:
        if not self.successful:
            return {"error": "No successful requests"}

        init_times = [r.init_time for r in self.successful]
        start_times = [r.start_run_time for r in self.successful]
        action_times = [r.action_time for r in self.successful]
        eval_times = [r.evaluate_time for r in self.successful]
        total_times = [r.total_time for r in self.successful]

        return {
            "total_requests": len(self.results),
            "successful": len(self.successful),
            "failed": len(self.failed),
            "init_environment": {
                "min": min(init_times),
                "max": max(init_times),
                "mean": statistics.mean(init_times),
                "median": statistics.median(init_times),
                "p95": sorted(init_times)[int(len(init_times) * 0.95)]
                if len(init_times) > 1
                else init_times[0],
            },
            "start_run": {
                "min": min(start_times),
                "max": max(start_times),
                "mean": statistics.mean(start_times),
                "median": statistics.median(start_times),
            },
            "action": {
                "min": min(action_times),
                "max": max(action_times),
                "mean": statistics.mean(action_times),
                "median": statistics.median(action_times),
            },
            "evaluate": {
                "min": min(eval_times),
                "max": max(eval_times),
                "mean": statistics.mean(eval_times),
                "median": statistics.median(eval_times),
            },
            "total": {
                "min": min(total_times),
                "max": max(total_times),
                "mean": statistics.mean(total_times),
                "median": statistics.median(total_times),
                "p95": sorted(total_times)[int(len(total_times) * 0.95)]
                if len(total_times) > 1
                else total_times[0],
            },
            "throughput_per_second": len(self.successful)
            / sum(total_times)
            * len(self.successful)
            if total_times
            else 0,
        }


class LoadTester:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def run_single_test(
        self, client: httpx.AsyncClient, test_id: int, template: str = "slack_default"
    ) -> TestResult:
        """Run a single end-to-end test."""
        env_id = ""
        run_id = ""

        try:
            total_start = time.perf_counter()

            # 1. Initialize environment
            t0 = time.perf_counter()
            init_resp = await client.post(
                f"{self.base_url}/api/platform/initEnv",
                json={
                    "templateSchema": template,
                    "ttlSeconds": 300,
                    "impersonateEmail": "agent@example.com",
                },
                timeout=self.timeout,
            )
            if init_resp.status_code >= 400:
                raise Exception(
                    f"initEnv failed: {init_resp.status_code} - {init_resp.text}"
                )
            init_resp.raise_for_status()
            init_data = init_resp.json()
            env_id = init_data["environmentId"]
            init_time = time.perf_counter() - t0
            print(
                f"  [{test_id}] init_environment: {init_time:.2f}s (env={env_id[:8]})"
            )

            # 2. Start test run
            t0 = time.perf_counter()
            start_resp = await client.post(
                f"{self.base_url}/api/platform/startRun",
                json={
                    "envId": env_id,
                },
                timeout=self.timeout,
            )
            if start_resp.status_code >= 400:
                raise Exception(
                    f"startRun failed: {start_resp.status_code} - {start_resp.text}"
                )
            start_resp.raise_for_status()
            start_data = start_resp.json()
            run_id = start_data["runId"]
            start_run_time = time.perf_counter() - t0
            print(f"  [{test_id}] start_run: {start_run_time:.2f}s (run={run_id[:8]})")

            # 3. Perform action - send a Slack message
            t0 = time.perf_counter()
            action_resp = await client.post(
                f"{self.base_url}/api/env/{env_id}/services/slack/chat.postMessage",
                json={
                    "channel": "general",
                    "text": f"Load test message #{test_id} at {time.time()}",
                },
                timeout=self.timeout,
            )
            action_resp.raise_for_status()
            action_time = time.perf_counter() - t0
            print(f"  [{test_id}] action (postMessage): {action_time:.2f}s")

            # Small delay to ensure replication captures the change
            await asyncio.sleep(0.5)

            # 4. Evaluate run (compute diff)
            t0 = time.perf_counter()
            eval_resp = await client.post(
                f"{self.base_url}/api/platform/evaluateRun",
                json={
                    "runId": run_id,
                },
                timeout=self.timeout,
            )
            eval_resp.raise_for_status()
            evaluate_time = time.perf_counter() - t0
            print(f"  [{test_id}] evaluate_run: {evaluate_time:.2f}s")

            total_time = time.perf_counter() - total_start
            print(f"  [{test_id}] TOTAL: {total_time:.2f}s ✓")

            return TestResult(
                env_id=env_id,
                run_id=run_id,
                init_time=init_time,
                start_run_time=start_run_time,
                action_time=action_time,
                evaluate_time=evaluate_time,
                total_time=total_time,
                success=True,
            )

        except Exception as e:
            print(f"  [{test_id}] FAILED: {e}")
            return TestResult(
                env_id=env_id,
                run_id=run_id,
                init_time=0,
                start_run_time=0,
                action_time=0,
                evaluate_time=0,
                total_time=0,
                success=False,
                error=str(e),
            )

    async def check_templates(self, client: httpx.AsyncClient) -> list[str]:
        """List available templates."""
        try:
            resp = await client.get(f"{self.base_url}/api/platform/templates")
            if resp.status_code == 200:
                data = resp.json()
                return [t.get("location") or t.get("name") for t in data]
        except Exception:
            pass
        return []

    async def run_load_test(
        self,
        num_requests: int,
        concurrency: int,
        template: str = "slack_default",
    ) -> LoadTestResults:
        """Run load test with specified concurrency."""
        results = LoadTestResults()
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_test(client: httpx.AsyncClient, test_id: int) -> TestResult:
            async with semaphore:
                return await self.run_single_test(client, test_id, template)

        print(f"\n{'=' * 60}")
        print(f"Load Test: {num_requests} requests, {concurrency} concurrent")
        print(f"Template: {template}")
        print(f"{'=' * 60}\n")

        overall_start = time.perf_counter()

        async with httpx.AsyncClient() as client:
            tasks = [bounded_test(client, i) for i in range(num_requests)]
            test_results = await asyncio.gather(*tasks)
            results.results.extend(test_results)

        overall_time = time.perf_counter() - overall_start

        print(f"\n{'=' * 60}")
        print(f"Completed in {overall_time:.2f}s")
        print(f"{'=' * 60}\n")

        return results


def print_summary(results: LoadTestResults):
    """Print formatted summary of results."""
    summary = results.summary()

    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)

    if summary.get("error"):
        print(
            f"\nRequests: {len(results.results)} total, "
            f"{len(results.successful)} successful, {len(results.failed)} failed"
        )
        print(f"\nError: {summary['error']}")
        if results.failed:
            print("\nFailure details:")
            for r in results.failed[:5]:  # Show first 5
                print(f"  - {r.error}")
        return

    print(
        f"\nRequests: {summary['total_requests']} total, "
        f"{summary['successful']} successful, {summary['failed']} failed"
    )

    print(f"\nThroughput: {summary['throughput_per_second']:.2f} req/s")

    for stage in ["init_environment", "start_run", "action", "evaluate", "total"]:
        data = summary[stage]
        print(f"\n{stage.upper()}:")
        print(f"  min: {data['min']:.3f}s")
        print(f"  max: {data['max']:.3f}s")
        print(f"  mean: {data['mean']:.3f}s")
        print(f"  median: {data['median']:.3f}s")
        if "p95" in data:
            print(f"  p95: {data['p95']:.3f}s")

    # Print failures if any
    if results.failed:
        print(f"\n{'=' * 60}")
        print("FAILURES:")
        for r in results.failed:
            print(f"  - {r.error}")


async def run_scaling_test(tester: LoadTester, max_concurrency: int = 10):
    """Run tests at increasing concurrency levels to see scaling behavior."""
    print("\n" + "=" * 60)
    print("SCALING TEST - How performance changes with concurrency")
    print("=" * 60 + "\n")

    scaling_results = []

    for concurrency in [1, 2, 5, 10, max_concurrency]:
        if concurrency > max_concurrency:
            break

        print(f"\n--- Concurrency: {concurrency} ---")
        results = await tester.run_load_test(
            num_requests=concurrency * 2,  # 2 requests per concurrent worker
            concurrency=concurrency,
        )
        summary = results.summary()
        scaling_results.append(
            {
                "concurrency": concurrency,
                "throughput": summary.get("throughput_per_second", 0),
                "mean_total": summary.get("total", {}).get("mean", 0),
                "p95_total": summary.get("total", {}).get("p95", 0),
                "success_rate": len(results.successful) / len(results.results) * 100
                if results.results
                else 0,
            }
        )

    print("\n" + "=" * 60)
    print("SCALING SUMMARY")
    print("=" * 60)
    print(
        f"\n{'Concurrency':<12} {'Throughput':<15} {'Mean Time':<12} {'P95 Time':<12} {'Success %':<10}"
    )
    print("-" * 60)
    for r in scaling_results:
        print(
            f"{r['concurrency']:<12} {r['throughput']:<15.2f} {r['mean_total']:<12.2f} {r['p95_total']:<12.2f} {r['success_rate']:<10.1f}"
        )


async def main():
    parser = argparse.ArgumentParser(description="Load test for diff-the-universe")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL of the API"
    )
    parser.add_argument(
        "--requests", type=int, default=10, help="Number of requests to make"
    )
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--scaling-test", action="store_true", help="Run scaling test instead"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Max concurrency for scaling test",
    )

    args = parser.parse_args()

    tester = LoadTester(base_url=args.base_url, timeout=args.timeout)

    if args.scaling_test:
        await run_scaling_test(tester, max_concurrency=args.max_concurrency)
    else:
        results = await tester.run_load_test(
            num_requests=args.requests,
            concurrency=args.concurrency,
        )
        print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
