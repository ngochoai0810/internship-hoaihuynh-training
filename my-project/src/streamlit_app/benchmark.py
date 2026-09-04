"""Asynchronous load-test engine used by the local demo dashboard."""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Pacing and connection limits for one benchmark run."""

    total_requests: int = 1000
    duration_seconds: float = 60.0
    concurrency: int = 20
    timeout_seconds: float = 10.0

    def validate(self) -> None:
        values = (
            self.total_requests,
            self.duration_seconds,
            self.concurrency,
            self.timeout_seconds,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Benchmark values must be greater than zero")


@dataclass(frozen=True)
class RequestOutcome:
    """Result and latency of one prediction request."""

    success: bool
    latency_ms: float
    error: str | None = None
    predicted_price: float | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate metrics displayed after a benchmark finishes."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate_percent: float
    elapsed_seconds: float
    throughput_rps: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    model_sha256: str
    error_counts: dict[str, int]
    predicted_price: float | None


def build_benchmark_result(
    outcomes: list[RequestOutcome],
    *,
    elapsed_seconds: float,
    model_sha256: str,
) -> BenchmarkResult:
    """Build deterministic aggregate metrics from individual outcomes."""

    if not outcomes:
        raise ValueError("At least one request outcome is required")
    if elapsed_seconds <= 0:
        raise ValueError("Elapsed time must be greater than zero")

    latencies = np.asarray([outcome.latency_ms for outcome in outcomes], dtype=float)
    successful = [outcome for outcome in outcomes if outcome.success]
    error_counts = Counter(
        outcome.error or "unknown_error" for outcome in outcomes if not outcome.success
    )
    predicted_price = next(
        (
            outcome.predicted_price
            for outcome in successful
            if outcome.predicted_price is not None
        ),
        None,
    )
    total = len(outcomes)
    success_count = len(successful)

    return BenchmarkResult(
        total_requests=total,
        successful_requests=success_count,
        failed_requests=total - success_count,
        error_rate_percent=(total - success_count) / total * 100,
        elapsed_seconds=elapsed_seconds,
        throughput_rps=total / elapsed_seconds,
        average_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        max_latency_ms=float(np.max(latencies)),
        model_sha256=model_sha256,
        error_counts=dict(error_counts),
        predicted_price=predicted_price,
    )


def _validate_prediction_response(
    response: httpx.Response,
    expected_model_sha256: str,
) -> tuple[str | None, float | None]:
    if response.status_code != 201:
        return f"http_{response.status_code}", None

    try:
        body: Any = response.json()
    except ValueError:
        return "invalid_response", None

    if not isinstance(body, dict):
        return "invalid_response", None
    if body.get("model_sha256") != expected_model_sha256:
        return "model_changed", None

    price = body.get("predicted_price")
    if (
        not isinstance(price, int | float)
        or isinstance(price, bool)
        or not math.isfinite(float(price))
        or float(price) < 0
        or body.get("currency") != "USD"
        or not isinstance(body.get("prediction_history_id"), int)
    ):
        return "invalid_response", None
    return None, float(price)


async def run_benchmark(
    *,
    api_url: str,
    token: str,
    payload: dict[str, Any],
    expected_model_sha256: str,
    config: BenchmarkConfig,
    progress_callback: ProgressCallback | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BenchmarkResult:
    """Send a fixed number of paced prediction requests and summarize them."""

    config.validate()
    start = time.perf_counter()
    interval = config.duration_seconds / config.total_requests
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0

    async with httpx.AsyncClient(
        base_url=api_url.rstrip("/"),
        timeout=config.timeout_seconds,
        transport=transport,
    ) as client:

        async def send_one(index: int) -> RequestOutcome:
            nonlocal completed
            target_time = start + index * interval
            wait_seconds = target_time - time.perf_counter()
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            async with semaphore:
                request_start = time.perf_counter()
                try:
                    response = await client.post(
                        "/api/v1/predict",
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    error, predicted_price = _validate_prediction_response(
                        response,
                        expected_model_sha256,
                    )
                except httpx.TimeoutException:
                    error, predicted_price = "timeout", None
                except httpx.RequestError:
                    error, predicted_price = "request_error", None
                latency_ms = (time.perf_counter() - request_start) * 1000
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, config.total_requests)
                return RequestOutcome(
                    success=error is None,
                    latency_ms=latency_ms,
                    error=error,
                    predicted_price=predicted_price,
                )

        outcomes = await asyncio.gather(
            *(send_one(index) for index in range(config.total_requests))
        )

    elapsed_seconds = time.perf_counter() - start
    return build_benchmark_result(
        list(outcomes),
        elapsed_seconds=elapsed_seconds,
        model_sha256=expected_model_sha256,
    )
