from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest


def test_prediction_comparison_reports_absolute_and_percentage_error() -> None:
    demo = import_module("streamlit_app.demo")

    comparison = demo.compare_prediction(
        predicted_price=229_452.05,
        actual_price=221_000.0,
    )

    assert comparison.absolute_error == pytest.approx(8_452.05)
    assert comparison.percentage_error == pytest.approx(3.824456, rel=1e-6)


def test_id_605_is_in_documented_retrain_holdout() -> None:
    source = pd.DataFrame({"Id": range(1, 1461)})
    subset = (
        source.sample(frac=0.6, random_state=2026).sort_index().reset_index(drop=True)
    )
    train, holdout = train_test_split(
        subset,
        test_size=0.2,
        random_state=42,
    )

    assert 605 not in train["Id"].values
    assert 605 in holdout["Id"].values


def test_evaluation_is_only_attributed_to_its_exact_model_hash() -> None:
    demo = import_module("streamlit_app.demo")

    evaluation = demo.get_verified_evaluation(demo.VERIFIED_MODEL_SHA256)

    assert evaluation is not None
    assert evaluation.sample_id == 605
    assert evaluation.metrics == {
        "RMSE log": pytest.approx(0.255751),
        "MAE log": pytest.approx(0.187704),
        "R²": pytest.approx(0.631583),
    }
    assert demo.get_verified_evaluation("b" * 64) is None


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
            "model_sha256": "a" * 64,
            "created_at": "2026-09-03T10:00:00",
        }
    ]

    rows = demo.format_history_rows(records)

    assert rows == [
        {
            "ID": 7,
            "Thời gian": "2026-09-03 10:00:00",
            "Giá dự đoán": "$229,452.05",
            "Mặt tiền": 88.0,
            "Tầng hầm": 1494.0,
            "Garage": "Attchd",
            "Chất lượng": "Gd",
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
from streamlit_app.demo import VERIFIED_MODEL_SHA256


class FakeClient:
    def predict(self, payload, token):
        return ApiResult(
            True,
            "ok",
            {
                "predicted_price": 229_452.05,
                "model_sha256": VERIFIED_MODEL_SHA256,
                "prediction_history_id": 1,
            },
        )


app.CLIENT = FakeClient()
health = ApiResult(True, "ok", {"model_sha256": VERIFIED_MODEL_SHA256})
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
    model_sha256=VERIFIED_MODEL_SHA256,
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
