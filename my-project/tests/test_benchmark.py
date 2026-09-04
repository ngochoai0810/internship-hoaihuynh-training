from __future__ import annotations

import asyncio

import httpx
import pytest
from streamlit_app.benchmark import (
    BenchmarkConfig,
    RequestOutcome,
    build_benchmark_result,
    run_benchmark,
)

VALID_PAYLOAD = {
    "lot_frontage": 88.0,
    "mas_vnr_area": 99.0,
    "total_bsmt_sf": 1494.0,
    "garage_type": "Attchd",
    "alley": None,
    "exter_qual": "Gd",
}
MODEL_SHA256 = "a" * 64


def test_build_benchmark_result_reports_hand_checked_metrics() -> None:
    outcomes = [
        RequestOutcome(True, 10.0, predicted_price=221_000.0),
        RequestOutcome(True, 20.0, predicted_price=221_000.0),
        RequestOutcome(False, 30.0, error="http_500"),
        RequestOutcome(False, 40.0, error="timeout"),
    ]

    result = build_benchmark_result(
        outcomes,
        elapsed_seconds=2.0,
        model_sha256=MODEL_SHA256,
    )

    assert result.total_requests == 4
    assert result.successful_requests == 2
    assert result.failed_requests == 2
    assert result.error_rate_percent == 50.0
    assert result.throughput_rps == 2.0
    assert result.average_latency_ms == 25.0
    assert result.p50_latency_ms == 25.0
    assert result.p95_latency_ms == pytest.approx(38.5)
    assert result.p99_latency_ms == pytest.approx(39.7)
    assert result.max_latency_ms == 40.0
    assert result.error_counts == {"http_500": 1, "timeout": 1}
    assert result.predicted_price == 221_000.0


def test_run_benchmark_sends_exact_total_and_reports_progress() -> None:
    request_count = 0
    progress_updates: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.url.path == "/api/v1/predict"
        assert request.headers["Authorization"] == "Bearer jwt-token"
        return httpx.Response(
            201,
            json={
                "prediction_history_id": request_count,
                "predicted_price": 229_452.05,
                "currency": "USD",
                "model_sha256": MODEL_SHA256,
            },
        )

    result = asyncio.run(
        run_benchmark(
            api_url="http://testserver",
            token="jwt-token",
            payload=VALID_PAYLOAD,
            expected_model_sha256=MODEL_SHA256,
            config=BenchmarkConfig(
                total_requests=5,
                duration_seconds=0.01,
                concurrency=2,
                timeout_seconds=1.0,
            ),
            progress_callback=lambda completed, total: progress_updates.append(
                (completed, total)
            ),
            transport=httpx.MockTransport(handler),
        )
    )

    assert request_count == 5
    assert result.successful_requests == 5
    assert result.failed_requests == 0
    assert result.error_rate_percent == 0.0
    assert result.predicted_price == 229_452.05
    assert progress_updates[-1] == (5, 5)


def test_run_benchmark_counts_http_model_and_payload_failures() -> None:
    responses = [
        httpx.Response(
            201,
            json={
                "prediction_history_id": 1,
                "predicted_price": 229_452.05,
                "currency": "USD",
                "model_sha256": MODEL_SHA256,
            },
        ),
        httpx.Response(500, json={"detail": "database locked"}),
        httpx.Response(
            201,
            json={
                "prediction_history_id": 2,
                "predicted_price": 229_452.05,
                "currency": "USD",
                "model_sha256": "b" * 64,
            },
        ),
        httpx.Response(201, content=b"not-json"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    result = asyncio.run(
        run_benchmark(
            api_url="http://testserver",
            token="jwt-token",
            payload=VALID_PAYLOAD,
            expected_model_sha256=MODEL_SHA256,
            config=BenchmarkConfig(
                total_requests=4,
                duration_seconds=0.01,
                concurrency=1,
                timeout_seconds=1.0,
            ),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.successful_requests == 1
    assert result.failed_requests == 3
    assert result.error_rate_percent == 75.0
    assert result.error_counts == {
        "http_500": 1,
        "model_changed": 1,
        "invalid_response": 1,
    }


@pytest.mark.parametrize(
    ("exception", "expected_error"),
    [
        (httpx.ReadTimeout("too slow"), "timeout"),
        (httpx.ConnectError("offline"), "request_error"),
    ],
)
def test_run_benchmark_records_transport_failures_without_stopping(
    exception: httpx.RequestError,
    expected_error: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception

    result = asyncio.run(
        run_benchmark(
            api_url="http://testserver",
            token="jwt-token",
            payload=VALID_PAYLOAD,
            expected_model_sha256=MODEL_SHA256,
            config=BenchmarkConfig(
                total_requests=1,
                duration_seconds=0.001,
                concurrency=1,
                timeout_seconds=1.0,
            ),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.failed_requests == 1
    assert result.error_counts == {expected_error: 1}


@pytest.mark.parametrize(
    "config",
    [
        BenchmarkConfig(total_requests=0),
        BenchmarkConfig(duration_seconds=0),
        BenchmarkConfig(concurrency=0),
        BenchmarkConfig(timeout_seconds=0),
    ],
)
def test_benchmark_config_rejects_non_positive_values(config: BenchmarkConfig) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        config.validate()
