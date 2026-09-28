"""Local FastAPI service; load trusted model once during application startup."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .inference import Predictor


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    timestamp: str
    speed: float = Field(ge=0, le=200)
    flow: StrictInt = Field(ge=1)
    speed_std: float = Field(ge=0)
    high_code_share: float = Field(ge=0, le=1)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation] = Field(min_length=4, max_length=4)


class ForecastResponse(BaseModel):
    prediction_time: str
    mean_speed_kmh: float
    horizon_minutes: int
    model_family: str
    seed: int


def create_app(artifact=None, family=None):
    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(1)
        path = artifact or os.environ.get("TRAFFIC_MODEL_PATH")
        if not path:
            raise RuntimeError("Set TRAFFIC_MODEL_PATH to a trusted local checkpoint")
        app.state.predictor = Predictor(
            Path(path), family or os.environ.get("TRAFFIC_MODEL_FAMILY", "gru")
        )
        yield
        del app.state.predictor

    app = FastAPI(
        title="One-minute traffic speed forecast", version="1.0", lifespan=lifespan
    )

    @app.get("/health")
    def health():
        return {"status": "ready", "model_family": app.state.predictor.family}

    @app.post("/predict", response_model=ForecastResponse)
    def predict(request: ForecastRequest):
        try:
            return app.state.predictor.predict(request.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
