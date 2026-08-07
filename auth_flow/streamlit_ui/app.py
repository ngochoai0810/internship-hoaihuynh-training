"""Minimal Streamlit UI for the auth_flow FastAPI backend."""

from __future__ import annotations

import requests
import streamlit as st

from config import API_URL, REQUEST_TIMEOUT

st.set_page_config(page_title="Auth Flow Demo", page_icon=":lock:")


def init_session_state() -> None:
    """Initialize session keys without resetting them on Streamlit reruns."""
    if "token" not in st.session_state:
        st.session_state["token"] = None
    if "email" not in st.session_state:
        st.session_state["email"] = None


def extract_error_detail(response: requests.Response) -> str:
    """Return FastAPI's error detail, with a fallback for non-JSON responses."""
    try:
        detail = response.json().get("detail", "Unknown error.")
    except ValueError:
        return f"Unknown error (status {response.status_code})."

    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def register_user(email: str, password: str) -> tuple[bool, str]:
    """Call POST /register with JSON."""
    payload = {"email": email, "password": password}
    try:
        response = requests.post(
            f"{API_URL}/register",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        return (
            False,
            f"Cannot connect to backend at {API_URL}. Is the FastAPI server running?",
        )

    if response.status_code in (200, 201):
        return True, "Registration successful. You can log in now."

    return False, extract_error_detail(response)


def login_user(email: str, password: str) -> tuple[bool, str]:
    """Call POST /login with OAuth2 form data."""
    form_data = {"username": email, "password": password}
    try:
        response = requests.post(
            f"{API_URL}/login",
            data=form_data,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        return (
            False,
            f"Cannot connect to backend at {API_URL}. Is the FastAPI server running?",
        )

    if response.status_code == 200:
        body = response.json()
        st.session_state["token"] = body.get("access_token")
        st.session_state["email"] = email
        return True, "Login successful."

    return False, extract_error_detail(response)


def get_current_user_profile() -> tuple[bool, str, dict | None]:
    """Call GET /me with the bearer token stored in Streamlit session state."""
    token = st.session_state.get("token")
    if not token:
        return (
            False,
            "No access token found. Please log in before calling /me.",
            None,
        )

    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(
            f"{API_URL}/me",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        return (
            False,
            f"Cannot connect to backend at {API_URL}. Is the FastAPI server running?",
            None,
        )

    if response.status_code == 200:
        return True, "Protected /me request successful.", response.json()

    if response.status_code == 401:
        return (
            False,
            "Unauthorized. The token is missing, invalid, or expired.",
            None,
        )

    return False, extract_error_detail(response), None


def render_auth_forms() -> None:
    """Render registration and login forms side by side."""
    col_register, col_login = st.columns(2)

    with col_register:
        st.subheader("Register")
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input(
                "Password",
                type="password",
                key="reg_password",
            )
            submitted = st.form_submit_button("Register")

        if submitted:
            if not reg_email or not reg_password:
                st.error("Please enter both email and password.")
            else:
                ok, message = register_user(reg_email, reg_password)
                (st.success if ok else st.error)(message)

    with col_login:
        st.subheader("Login")
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )
            submitted = st.form_submit_button("Login")

        if submitted:
            if not login_email or not login_password:
                st.error("Please enter both email and password.")
            else:
                ok, message = login_user(login_email, login_password)
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


def render_logged_in_view() -> None:
    """Render logged-in state, protected profile lookup, and logout action."""
    st.success(f"Logged in as: **{st.session_state['email']}**")

    token = st.session_state["token"]
    if token:
        st.caption(f"Access token: `{token[:20]}...`")

    if st.button("Call /me"):
        ok, message, profile = get_current_user_profile()
        if ok and profile is not None:
            st.success(message)
            st.json(profile)
        else:
            st.error(message)

    if st.button("Logout"):
        st.session_state["token"] = None
        st.session_state["email"] = None
        st.rerun()


def main() -> None:
    st.title("Auth Flow Demo")
    st.caption(f"Backend API: {API_URL}")

    init_session_state()

    if st.session_state["token"]:
        render_logged_in_view()
    else:
        render_auth_forms()


if __name__ == "__main__":
    main()
