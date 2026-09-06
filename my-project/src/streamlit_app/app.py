"""Streamlit UI for authentication and live house-price inference."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit_app.client import ApiClient

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
CLIENT = ApiClient(API_URL, timeout=REQUEST_TIMEOUT)


def initialize_session() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("email", None)


def render_authentication() -> None:
    register_tab, login_tab = st.tabs(["Register", "Login"])
    with register_tab:
        with st.form("register"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Password", type="password", key="register_password"
            )
            submitted = st.form_submit_button("Register")
        if submitted:
            result = CLIENT.register(email, password)
            (st.success if result.ok else st.error)(result.message)

    with login_tab:
        with st.form("login"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
        if submitted:
            result = CLIENT.login(email, password)
            if result.ok and result.data is not None:
                st.session_state["token"] = result.data["access_token"]
                st.session_state["email"] = email
                st.rerun()
            st.error(result.message)


def render_prediction() -> None:
    st.success(f"Logged in as {st.session_state['email']}")
    with st.form("prediction"):
        lot_frontage = st.number_input("Lot frontage", min_value=0.0, value=70.0)
        mas_vnr_area = st.number_input(
            "Masonry veneer area", min_value=0.0, value=100.0
        )
        total_bsmt_sf = st.number_input(
            "Total basement area", min_value=0.0, value=856.0
        )
        garage_type = st.selectbox(
            "Garage type", ["Attchd", "Detchd", "BuiltIn", "None"]
        )
        alley = st.selectbox("Alley", ["None", "Grvl", "Pave"])
        exter_qual = st.selectbox("Exterior quality", ["Po", "Fa", "TA", "Gd", "Ex"])
        submitted = st.form_submit_button("Estimate price")

    if submitted:
        result = CLIENT.predict(
            {
                "lot_frontage": lot_frontage,
                "mas_vnr_area": mas_vnr_area,
                "total_bsmt_sf": total_bsmt_sf,
                "garage_type": garage_type,
                "alley": alley,
                "exter_qual": exter_qual,
            },
            token=st.session_state["token"],
        )
        if result.ok and result.data is not None:
            st.metric("Estimated price", f"${result.data['predicted_price']:,.2f}")
            st.caption(f"Model: {result.data['model_sha256'][:12]}")
        else:
            st.error(result.message)

    if st.button("Logout"):
        st.session_state["token"] = None
        st.session_state["email"] = None
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="House Price Estimator", page_icon="🏠")
    st.title("House Price Estimator")
    st.caption(f"Backend: {API_URL}")
    initialize_session()
    if st.session_state["token"]:
        render_prediction()
    else:
        render_authentication()


if __name__ == "__main__":
    main()
