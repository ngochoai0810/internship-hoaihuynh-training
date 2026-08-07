"""
App entrypoint for testing the auth flow through /docs:
register -> login -> Authorize -> /me.
"""

from fastapi import FastAPI

from src.api.routes.auth import router as auth_router

app = FastAPI(title="Auth Flow - Day 26-28")

app.include_router(auth_router, tags=["auth"])
