from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

MODEL_SHA256 = "a" * 64
DEMO_SAMPLE = {
    "index": 604,
    "id": 605,
    "actual_price": 221_000.0,
    "payload": {
        "overall_qual": 8,
        "gr_liv_area": 1750.0,
        "garage_cars": 2.0,
        "garage_area": 550.0,
        "total_bsmt_sf": 1494.0,
        "first_flr_sf": 1494.0,
        "full_bath": 2,
        "tot_rms_abv_grd": 7,
        "year_built": 2005,
        "year_remod_add": 2006,
        "neighborhood": "CollgCr",
        "garage_type": "Attchd",
        "exter_qual": "Gd",
        "kitchen_qual": "Gd",
        "bsmt_qual": "Gd",
    },
}
FINAL_REPORT: dict[str, Any] = {
    "model_name": "ridge",
    "model_sha256": MODEL_SHA256,
    "data_path": "my-project/data/raw/train.csv",
    "test_size": 0.2,
    "random_state": 42,
    "metrics": {
        "rmse_log": 0.255751,
        "mae_log": 0.187704,
        "r2_log": 0.631583,
        "rmse_price": 41_000.0,
        "mae_price": 27_000.0,
        "r2_price": 0.61,
    },
}


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_prediction_comparison_reports_absolute_and_percentage_error() -> None:
    demo = import_module("streamlit_app.demo")

    comparison = demo.compare_prediction(
        predicted_price=229_452.05,
        actual_price=221_000.0,
    )

    assert comparison.absolute_error == pytest.approx(8_452.05)
    assert comparison.percentage_error == pytest.approx(3.824456, rel=1e-6)


def test_holdout_sample_comes_from_the_finalize_artifact(tmp_path: Path) -> None:
    demo = import_module("streamlit_app.demo")
    samples_path = _write_json(tmp_path / "demo_samples.json", [DEMO_SAMPLE])

    sample = demo.load_holdout_sample(samples_path)

    assert sample is not None
    assert sample.sample_id == 605
    assert sample.actual_price == pytest.approx(221_000.0)
    assert sample.payload == DEMO_SAMPLE["payload"]


def test_missing_finalize_artifact_yields_no_holdout_sample(tmp_path: Path) -> None:
    demo = import_module("streamlit_app.demo")

    assert demo.load_holdout_sample(tmp_path / "absent.json") is None


def test_verified_evaluation_reads_final_holdout_metrics(tmp_path: Path) -> None:
    demo = import_module("streamlit_app.demo")
    report_path = _write_json(tmp_path / "final_holdout_report.json", FINAL_REPORT)

    evaluation = demo.load_verified_evaluation(report_path)

    assert evaluation is not None
    assert evaluation.model_sha256 == MODEL_SHA256
    assert evaluation.metrics == {
        "RMSE log": pytest.approx(0.255751),
        "MAE log": pytest.approx(0.187704),
        "R²": pytest.approx(0.631583),
    }


def test_evaluation_is_only_attributed_to_its_exact_model_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = import_module("streamlit_app.demo")
    report_path = _write_json(tmp_path / "final_holdout_report.json", FINAL_REPORT)
    monkeypatch.setattr(
        demo,
        "VERIFIED_MODEL_EVALUATION",
        demo.load_verified_evaluation(report_path),
    )

    assert demo.get_verified_evaluation(MODEL_SHA256) is not None
    assert demo.get_verified_evaluation("b" * 64) is None


def test_no_evaluation_is_claimed_before_the_model_is_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = import_module("streamlit_app.demo")
    monkeypatch.setattr(demo, "VERIFIED_MODEL_EVALUATION", None)

    assert demo.get_verified_evaluation(MODEL_SHA256) is None


def test_demo_load_test_count_is_fixed_at_one_thousand() -> None:
    demo = import_module("streamlit_app.demo")

    assert demo.DEMO_TOTAL_REQUESTS == 1000


def test_history_rows_are_flat_and_readable() -> None:
    demo = import_module("streamlit_app.demo")
    records = [
        {
            "id": 7,
            "input_payload": {
                "overall_qual": 8,
                "gr_liv_area": 1750.0,
                "total_bsmt_sf": 1494.0,
                "neighborhood": "CollgCr",
                "garage_type": "Attchd",
                "exter_qual": "Gd",
            },
            "predicted_price": 229_452.05,
            "model_sha256": MODEL_SHA256,
            "created_at": "2026-09-03T10:00:00",
        }
    ]

    rows = demo.format_history_rows(records)

    assert rows == [
        {
            "ID": 7,
            "Thời gian": "2026-09-03 10:00:00",
            "Giá dự đoán": "$229,452.05",
            "Chất lượng chung": 8,
            "Diện tích ở": 1750.0,
            "Tầng hầm": 1494.0,
            "Khu vực": "CollgCr",
            "Garage": "Attchd",
            "Model": "aaaaaaaaaaaa",
        }
    ]


def test_streamlit_login_page_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "streamlit_app" / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "House Price Prediction"
    assert [tab.label for tab in app.tabs] == ["Đăng ký", "Đăng nhập"]


def test_dashboard_omits_removed_explanatory_content() -> None:
    app = AppTest.from_string("""
from streamlit_app import app
from streamlit_app.benchmark import BenchmarkResult
from streamlit_app.client import ApiResult
MODEL_SHA256 = "a" * 64


class FakeClient:
    def predict(self, payload, token):
        return ApiResult(
            True,
            "ok",
            {
                "predicted_price": 229_452.05,
                "model_sha256": MODEL_SHA256,
                "prediction_history_id": 1,
            },
        )


app.CLIENT = FakeClient()
health = ApiResult(True, "ok", {"model_sha256": MODEL_SHA256})
benchmark = BenchmarkResult(
    total_requests=1000,
    successful_requests=1000,
    failed_requests=0,
    error_rate_percent=0.0,
    elapsed_seconds=60.0,
    throughput_rps=16.67,
    average_latency_ms=20.0,
    p50_latency_ms=18.0,
    p95_latency_ms=30.0,
    p99_latency_ms=40.0,
    max_latency_ms=50.0,
    model_sha256=MODEL_SHA256,
    error_counts={},
    predicted_price=229_452.05,
)

app.initialize_session()
app.render_overview(health)
app.render_prediction(health)
app.render_load_test(health)
app._display_benchmark_result(benchmark)
""").run(timeout=10)

    assert not app.exception
    assert all(
        element.value != "Hệ thống hoạt động như thế nào?" for element in app.subheader
    )
    assert all("Người dùng\n" not in element.value for element in app.code)
    assert all(
        "Streamlit là màn hình người dùng nhìn thấy" not in element.value
        for element in app.markdown
    )
    assert all(
        "Với đúng model hash đang chạy" not in element.value for element in app.info
    )
    assert all(
        "1.000 request trong 60 giây" not in element.value for element in app.info
    )
    assert all(
        "Tất cả response đều có status 201" not in element.value
        for element in app.success
    )

    prediction_submit = next(
        button for button in app.button if button.label == "Dự đoán giá"
    )
    prediction_submit.click().run(timeout=10)

    assert not app.exception
    assert all(
        "Sai số của một căn nhà không phải độ chính xác của toàn model"
        not in element.value
        for element in app.warning
    )


def test_manual_defaults_cover_the_whole_prediction_schema() -> None:
    demo = import_module("streamlit_app.demo")
    schema = import_module("schemas.prediction")

    assert set(demo.MANUAL_DEFAULT_PAYLOAD) == set(schema.PredictionInput.model_fields)
    schema.PredictionInput(**demo.MANUAL_DEFAULT_PAYLOAD)


def test_submitted_form_payload_passes_api_validation(tmp_path: Path) -> None:
    captured = tmp_path / "payload.json"
    app = AppTest.from_string(f"""
import json

from streamlit_app import app
from streamlit_app.client import ApiResult


class CapturingClient:
    def predict(self, payload, token):
        with open({str(captured)!r}, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return ApiResult(
            True,
            "ok",
            {{
                "predicted_price": 229_452.05,
                "model_sha256": {MODEL_SHA256!r},
                "prediction_history_id": 1,
            }},
        )


app.CLIENT = CapturingClient()
app.initialize_session()
app.render_prediction(ApiResult(True, "ok", {{"model_sha256": {MODEL_SHA256!r}}}))
""").run(timeout=10)

    assert not app.exception
    next(button for button in app.button if button.label == "Dự đoán giá").click().run(
        timeout=10
    )

    assert not app.exception
    schema = import_module("schemas.prediction")
    payload = json.loads(captured.read_text(encoding="utf-8"))
    assert set(payload) == set(schema.PredictionInput.model_fields)
    schema.PredictionInput(**payload)
