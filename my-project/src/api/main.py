"""FastAPI application factory with model-loading lifespan."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from api.routes import auth, prediction
from core.config import Settings, get_settings
from database import create_database
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml.runtime import load_model_artifact
from schemas.prediction import HealthResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independently configurable application instance."""

    app_settings = settings or get_settings()

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

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(auth.router, prefix=app_settings.api_v1_prefix)
    application.include_router(prediction.router, prefix=app_settings.api_v1_prefix)

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=application.state.model is not None,
            model_sha256=application.state.model_sha256,
        )

    return application


app = create_app()
