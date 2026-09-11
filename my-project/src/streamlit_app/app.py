"""Vietnamese Streamlit dashboard for the end-to-end technical demo."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from streamlit_app.benchmark import BenchmarkConfig, BenchmarkResult, run_benchmark
from streamlit_app.client import ApiClient, ApiResult
from streamlit_app.demo import (
    BASEMENT_QUALITY_OPTIONS,
    BENCHMARK_PAYLOAD,
    DEMO_TOTAL_REQUESTS,
    GARAGE_TYPE_OPTIONS,
    HOLDOUT_SAMPLE,
    MANUAL_DEFAULT_PAYLOAD,
    NEIGHBORHOOD_OPTIONS,
    QUALITY_OPTIONS,
    compare_prediction,
    format_history_rows,
    get_verified_evaluation,
)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
CLIENT = ApiClient(API_URL, timeout=REQUEST_TIMEOUT)


def initialize_session() -> None:
    """Create stable keys used across Streamlit reruns."""

    st.session_state.setdefault("token", None)
    st.session_state.setdefault("email", None)
    st.session_state.setdefault("benchmark_result", None)


def render_authentication() -> None:
    """Render registration and login without exposing protected demo tabs."""

    register_tab, login_tab = st.tabs(["Đăng ký", "Đăng nhập"])

    with register_tab:
        with st.form("register"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Mật khẩu (ít nhất 8 ký tự)",
                type="password",
                key="register_password",
            )
            submitted = st.form_submit_button("Tạo tài khoản")
        if submitted:
            result = CLIENT.register(email, password)
            if result.ok:
                st.success("Đăng ký thành công. Hãy chuyển sang tab Đăng nhập.")
            else:
                st.error(result.message)

    with login_tab:
        with st.form("login"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Mật khẩu", type="password", key="login_password")
            submitted = st.form_submit_button("Đăng nhập")
        if submitted:
            result = CLIENT.login(email, password)
            if result.ok and result.data is not None:
                st.session_state["token"] = result.data["access_token"]
                st.session_state["email"] = email
                st.rerun()
            else:
                st.error(result.message)


def render_status_header(health: ApiResult) -> None:
    """Show the three pieces of state most useful during a live demo."""

    user_column, backend_column, model_column, logout_column = st.columns([2, 1, 2, 1])
    user_column.metric("Người dùng", str(st.session_state["email"]))
    backend_column.metric("Backend", "Sẵn sàng" if health.ok else "Mất kết nối")
    model_hash = "Không có"
    if health.ok and health.data is not None:
        model_hash = str(health.data.get("model_sha256", "Không có"))[:12]
    model_column.metric("Dấu vân tay model", model_hash)
    if logout_column.button("Đăng xuất", use_container_width=True):
        st.session_state["token"] = None
        st.session_state["email"] = None
        st.session_state["benchmark_result"] = None
        st.rerun()


def render_overview(health: ApiResult) -> None:
    """Show backend and model health details."""

    if health.ok and health.data is not None:
        st.success("Backend đang hoạt động và model đã được nạp vào RAM.")
        st.json(health.data)
    else:
        st.error(
            f"Không kết nối được backend tại {API_URL}. "
            "Hãy khởi động Uvicorn rồi làm mới."
        )
    if st.button("Làm mới trạng thái"):
        st.rerun()


def _prediction_payload(use_holdout: bool) -> dict[str, Any] | None:
    """Render the prediction form and return values only after submission."""

    defaults: dict[str, Any] = (
        HOLDOUT_SAMPLE.payload
        if use_holdout and HOLDOUT_SAMPLE is not None
        else MANUAL_DEFAULT_PAYLOAD
    )

    def _option_index(options: list[str], key: str) -> int:
        value = defaults.get(key)
        return options.index(str(value)) if str(value) in options else 0

    with st.form("prediction"):
        with st.expander("Diện tích và phòng", expanded=True):
            left, right = st.columns(2)
            with left:
                gr_liv_area = st.number_input(
                    "Diện tích ở trên mặt đất (sq ft)",
                    min_value=0.0,
                    value=float(defaults["gr_liv_area"]),
                    disabled=use_holdout,
                )
                first_flr_sf = st.number_input(
                    "Diện tích tầng một (sq ft)",
                    min_value=0.0,
                    value=float(defaults["first_flr_sf"]),
                    disabled=use_holdout,
                )
                total_bsmt_sf = st.number_input(
                    "Tổng diện tích tầng hầm (sq ft)",
                    min_value=0.0,
                    value=float(defaults["total_bsmt_sf"]),
                    disabled=use_holdout,
                )
            with right:
                full_bath = st.number_input(
                    "Số phòng tắm đầy đủ",
                    min_value=0,
                    step=1,
                    value=int(defaults["full_bath"]),
                    disabled=use_holdout,
                )
                tot_rms_abv_grd = st.number_input(
                    "Số phòng trên mặt đất",
                    min_value=0,
                    step=1,
                    value=int(defaults["tot_rms_abv_grd"]),
                    disabled=use_holdout,
                )

        with st.expander("Năm và chất lượng", expanded=True):
            left, right = st.columns(2)
            with left:
                year_built = st.number_input(
                    "Năm xây dựng",
                    min_value=0,
                    step=1,
                    value=int(defaults["year_built"]),
                    disabled=use_holdout,
                )
                year_remod_add = st.number_input(
                    "Năm cải tạo",
                    min_value=0,
                    step=1,
                    value=int(defaults["year_remod_add"]),
                    disabled=use_holdout,
                )
                overall_qual = st.number_input(
                    "Chất lượng tổng thể (1-10)",
                    min_value=0,
                    max_value=10,
                    step=1,
                    value=int(defaults["overall_qual"]),
                    disabled=use_holdout,
                )
            with right:
                exter_qual = st.selectbox(
                    "Chất lượng bên ngoài",
                    QUALITY_OPTIONS,
                    index=_option_index(QUALITY_OPTIONS, "exter_qual"),
                    disabled=use_holdout,
                )
                kitchen_qual = st.selectbox(
                    "Chất lượng nhà bếp",
                    QUALITY_OPTIONS,
                    index=_option_index(QUALITY_OPTIONS, "kitchen_qual"),
                    disabled=use_holdout,
                )
                bsmt_qual = st.selectbox(
                    "Chất lượng tầng hầm",
                    BASEMENT_QUALITY_OPTIONS,
                    index=_option_index(BASEMENT_QUALITY_OPTIONS, "bsmt_qual"),
                    disabled=use_holdout,
                )

        with st.expander("Garage và khu vực", expanded=True):
            left, right = st.columns(2)
            with left:
                garage_cars = st.number_input(
                    "Số chỗ đậu xe",
                    min_value=0.0,
                    value=float(defaults["garage_cars"]),
                    disabled=use_holdout,
                )
                garage_area = st.number_input(
                    "Diện tích garage (sq ft)",
                    min_value=0.0,
                    value=float(defaults["garage_area"]),
                    disabled=use_holdout,
                )
            with right:
                garage_type = st.selectbox(
                    "Loại garage",
                    GARAGE_TYPE_OPTIONS,
                    index=_option_index(GARAGE_TYPE_OPTIONS, "garage_type"),
                    disabled=use_holdout,
                )
                neighborhood = st.selectbox(
                    "Khu vực",
                    NEIGHBORHOOD_OPTIONS,
                    index=_option_index(NEIGHBORHOOD_OPTIONS, "neighborhood"),
                    disabled=use_holdout,
                )

        submitted = st.form_submit_button(
            "Dự đoán giá", type="primary", use_container_width=True
        )

    if not submitted:
        return None
    return {
        "overall_qual": int(overall_qual),
        "gr_liv_area": float(gr_liv_area),
        "garage_cars": float(garage_cars),
        "garage_area": float(garage_area),
        "total_bsmt_sf": float(total_bsmt_sf),
        "first_flr_sf": float(first_flr_sf),
        "full_bath": int(full_bath),
        "tot_rms_abv_grd": int(tot_rms_abv_grd),
        "year_built": int(year_built),
        "year_remod_add": int(year_remod_add),
        "neighborhood": neighborhood,
        "garage_type": None if garage_type == "None" else garage_type,
        "exter_qual": exter_qual,
        "kitchen_qual": kitchen_qual,
        "bsmt_qual": bsmt_qual,
    }


def render_prediction(health: ApiResult) -> None:
    """Run one prediction and optionally compare it with a known target."""

    st.subheader("Dự đoán và đối chiếu đáp án")
    health_hash = ""
    if health.ok and health.data is not None:
        health_hash = str(health.data.get("model_sha256", ""))
    verified_evaluation = get_verified_evaluation(health_hash)

    use_reference_sample = False
    if HOLDOUT_SAMPLE is None:
        st.info(
            "Chưa có mẫu holdout. Chạy `python -m ml.finalize` để sinh "
            "`artifacts/final/demo_samples.json` rồi mở lại trang này."
        )
    else:
        sample_label = (
            f"Mẫu holdout — Id {HOLDOUT_SAMPLE.sample_id}"
            if verified_evaluation is not None
            else f"Mẫu tham chiếu — Id {HOLDOUT_SAMPLE.sample_id}"
        )
        sample_mode = st.radio(
            "Nguồn dữ liệu",
            [sample_label, "Nhập thủ công"],
            horizontal=True,
        )
        use_reference_sample = not sample_mode.startswith("Nhập thủ công")
        if use_reference_sample and verified_evaluation is None:
            st.warning(
                "Model hiện tại không khớp artifact đã xác minh. "
                f"Id {HOLDOUT_SAMPLE.sample_id} chỉ được dùng làm mẫu tham chiếu "
                f"có SalePrice ${HOLDOUT_SAMPLE.actual_price:,.0f}, không được "
                "gọi là holdout của model này."
            )
    # Lấy token từ session
    payload = _prediction_payload(use_reference_sample)
    if payload is None:
        return

    result = CLIENT.predict(payload, token=str(st.session_state["token"]))
    if not result.ok or result.data is None:
        st.error(result.message)
        return

    predicted_price = float(result.data["predicted_price"])
    model_hash = str(result.data["model_sha256"])
    if use_reference_sample and HOLDOUT_SAMPLE is not None:
        comparison = compare_prediction(
            predicted_price=predicted_price,
            actual_price=HOLDOUT_SAMPLE.actual_price,
        )
        actual_column, predicted_column, difference_column, error_column = st.columns(4)
        actual_column.metric("Giá thật", f"${HOLDOUT_SAMPLE.actual_price:,.0f}")
        predicted_column.metric("Giá dự đoán", f"${predicted_price:,.2f}")
        difference_column.metric(
            "Chênh lệch tuyệt đối", f"${comparison.absolute_error:,.2f}"
        )
        error_column.metric(
            "Sai số mẫu tham chiếu", f"{comparison.percentage_error:.2f}%"
        )
        response_evaluation = get_verified_evaluation(model_hash)
        if response_evaluation is not None:
            st.caption(
                " "
                + " | ".join(
                    f"{name}: {value:.3f}"
                    for name, value in response_evaluation.metrics.items()
                )
            )
        else:
            st.warning(
                "Chưa có metadata đánh giá khớp model hash này. Chênh lệch trên "
                "chỉ là so sánh tham chiếu; không hiển thị lại metrics của model cũ."
            )
    else:
        st.metric("Giá dự đoán", f"${predicted_price:,.2f}")
    st.caption(
        f"Prediction history ID: {result.data['prediction_history_id']} · "
        f"Model SHA-256: {model_hash}"
    )


def _display_benchmark_result(result: BenchmarkResult) -> None:
    """Render availability and latency metrics without calling them accuracy."""

    success_column, failed_column, error_column, throughput_column = st.columns(4)
    success_column.metric("Thành công", f"{result.successful_requests:,}")
    failed_column.metric("Thất bại", f"{result.failed_requests:,}")
    error_column.metric("Tỷ lệ lỗi", f"{result.error_rate_percent:.2f}%")
    throughput_column.metric("Thông lượng", f"{result.throughput_rps:.2f} req/s")

    st.table(
        {
            "Chỉ số": ["Trung bình", "p50", "p95", "p99", "Lớn nhất"],
            "Độ trễ (ms)": [
                round(result.average_latency_ms, 2),
                round(result.p50_latency_ms, 2),
                round(result.p95_latency_ms, 2),
                round(result.p99_latency_ms, 2),
                round(result.max_latency_ms, 2),
            ],
        }
    )
    st.caption(
        f"Đã gửi {result.total_requests:,} request trong "
        f"{result.elapsed_seconds:.2f}s · "
        f"Model {result.model_sha256[:12]}"
    )
    if result.failed_requests != 0:
        st.error("Có request thất bại. Chi tiết được nhóm theo nguyên nhân bên dưới.")
        st.json(result.error_counts)


def render_load_test(health: ApiResult) -> None:
    """Configure and execute a paced local load test."""

    st.subheader("Load test API dự đoán")
    with st.form("load_test"):
        request_column, duration_column, concurrency_column, timeout_column = (
            st.columns(4)
        )
        request_column.metric("Tổng request", f"{DEMO_TOTAL_REQUESTS:,} (cố định)")
        duration_seconds = duration_column.number_input(
            "Thời gian (giây)", min_value=1.0, value=60.0, step=1.0
        )
        concurrency = concurrency_column.number_input(
            "Concurrency", min_value=1, max_value=100, value=20, step=1
        )
        timeout_seconds = timeout_column.number_input(
            "Timeout (giây)", min_value=1.0, value=10.0, step=1.0
        )
        submitted = st.form_submit_button(
            "Bắt đầu load test",
            type="primary",
            disabled=not health.ok,
            use_container_width=True,
        )

    if submitted:
        if health.data is None:
            st.error("Backend chưa sẵn sàng.")
            return
        progress = st.progress(0.0, text="Đang gửi request...")
        last_percent = -1

        def update_progress(completed: int, total: int) -> None:
            nonlocal last_percent
            percent = int(completed / total * 100)
            if percent != last_percent:
                last_percent = percent
                progress.progress(
                    completed / total,
                    text=f"Đã hoàn thành {completed:,}/{total:,} request",
                )

        try:
            benchmark = asyncio.run(
                run_benchmark(
                    api_url=API_URL,
                    token=str(st.session_state["token"]),
                    payload=dict(BENCHMARK_PAYLOAD),
                    expected_model_sha256=str(health.data["model_sha256"]),
                    config=BenchmarkConfig(
                        total_requests=DEMO_TOTAL_REQUESTS,
                        duration_seconds=float(duration_seconds),
                        concurrency=int(concurrency),
                        timeout_seconds=float(timeout_seconds),
                    ),
                    progress_callback=update_progress,
                )
            )
        except (KeyError, ValueError) as exc:
            st.error(f"Không thể chạy load test: {exc}")
        else:
            st.session_state["benchmark_result"] = benchmark
            progress.progress(1.0, text="Đã hoàn thành load test")

    stored_result = st.session_state.get("benchmark_result")
    if isinstance(stored_result, BenchmarkResult):
        _display_benchmark_result(stored_result)


def render_history() -> None:
    """Display only the authenticated user's latest prediction records."""

    st.subheader("Lịch sử dự đoán của tài khoản hiện tại")
    limit = st.slider("Số bản ghi", min_value=1, max_value=100, value=20)
    result = CLIENT.prediction_history(
        token=str(st.session_state["token"]), limit=limit
    )
    if not result.ok or result.data is None:
        st.error(result.message)
        return

    records = result.data.get("items", [])
    rows = format_history_rows(records)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có lịch sử. Hãy chạy một prediction trước.")


def render_dashboard() -> None:
    """Render the authenticated four-tab presentation dashboard."""

    health = CLIENT.health()
    render_status_header(health)
    overview_tab, prediction_tab, load_tab, history_tab = st.tabs(
        ["Tổng quan", "Dự đoán & đối chiếu", "Load test", "Lịch sử"]
    )
    with overview_tab:
        render_overview(health)
    with prediction_tab:
        render_prediction(health)
    with load_tab:
        render_load_test(health)
    with history_tab:
        render_history()


def main() -> None:
    """Configure and render the application."""

    st.set_page_config(
        page_title="House Price Prediction",
        page_icon="",
        layout="wide",
    )
    st.title("House Price Prediction")
    st.caption(f"Frontend Streamlit · Backend {API_URL}")
    initialize_session()
    if st.session_state["token"]:
        render_dashboard()
    else:
        render_authentication()


if __name__ == "__main__":
    main()
