"""Configuration for the Streamlit UI."""

import os

from dotenv import load_dotenv

load_dotenv()

API_URL: str = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
