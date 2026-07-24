from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import houses, predict, users
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(houses.router, prefix=settings.api_v1_prefix)
app.include_router(predict.router, prefix=settings.api_v1_prefix)
app.include_router(users.router, prefix=settings.api_v1_prefix)
app.include_router(houses.router)
app.include_router(predict.router)
app.include_router(users.router)


@app.get("/", tags=["root"])
def read_root() -> dict[str, str]:
    return {"message": "House Price API"}


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
