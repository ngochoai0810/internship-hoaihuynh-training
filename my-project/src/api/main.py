"""FastAPI application factory with model-loading lifespan."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from api.routes import auth, demo, prediction
from api.routes.auth import ensure_demo_account
from core.config import Settings, get_settings
from core.logging import configure_logging
from database import create_database
from fastapi import FastAPI, Request
from models import Base
from fastapi.middleware.cors import CORSMiddleware
from ml.runtime import load_model_artifact
from schemas.prediction import HealthResponse
from starlette.responses import Response

REQUEST_LOGGER = logging.getLogger("api.request")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independently configurable application instance."""

    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        model, fingerprint = load_model_artifact(app_settings.model_path)
        application.state.model = model
        application.state.model_sha256 = fingerprint
        yield
        application.state.model = None
        application.state.engine.dispose()

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        lifespan=lifespan,
    )
    engine, session_factory = create_database(app_settings.database_url)
    application.state.settings = app_settings
    application.state.engine = engine
    application.state.session_factory = session_factory

    Base.metadata.create_all(bind=engine)
    with session_factory() as session:
        ensure_demo_account(session)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(auth.router, prefix=app_settings.api_v1_prefix)
    application.include_router(demo.router, prefix=app_settings.api_v1_prefix)
    application.include_router(prediction.router, prefix=app_settings.api_v1_prefix)

    @application.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1000
            REQUEST_LOGGER.exception(
                "request_id=%s method=%s path=%s status_code=500 duration_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id
        REQUEST_LOGGER.info(
            "request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=application.state.model is not None,
            model_sha256=application.state.model_sha256,
        )

    return application


app = create_app()
