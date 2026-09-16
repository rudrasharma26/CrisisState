"""Main FastAPI application entry point for CrisisState."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from crisisstate.app.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="CrisisState Core API",
        description="Uncertainty-aware incident intelligence system for disaster-response analysts (Urban Flooding)",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/health")
    def health():
        return {"status": "ok", "system": "CrisisState"}

    return app


app = create_app()
