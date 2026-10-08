"""FastAPI service exposing the trained fraud model with its rule-engine baseline."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from adshield.api.schemas import (
    MAX_BATCH,
    BatchScoreResponse,
    ClickRequest,
    FeatureContribution,
    HealthResponse,
    ScoreResponse,
)
from adshield.api.service import ScoredClick, ScoringService
from adshield.models.classifier import MODELS_DIR, ModelConfig, TrainedModel, load_model
from adshield.models.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

MODEL_DIR_ENV = "ADSHIELD_MODEL_DIR"


def _latest_model_dir(models_dir: Path) -> Path | None:
    candidates = [p for p in models_dir.glob("*/metrics.json")]
    return max(candidates, key=lambda p: p.stat().st_mtime).parent if candidates else None


def _load_startup_model(model_dir: Path | None) -> TrainedModel | None:
    """Load the model once. An explicitly requested model that fails to load aborts startup;
    with nothing requested and nothing found, the service runs rules-only (health says so)."""
    explicit = model_dir or (Path(os.environ[MODEL_DIR_ENV]) if os.environ.get(MODEL_DIR_ENV) else None)
    if explicit is not None:
        return load_model(explicit)  # raises: never silently serve a different model than asked for
    found = _latest_model_dir(MODELS_DIR)
    if found is None:
        logger.warning("no trained model found in %s; serving rules-only", MODELS_DIR)
        return None
    return load_model(found)


def get_service(request: Request) -> ScoringService:
    return request.app.state.service


Service = Annotated[ScoringService, Depends(get_service)]


def create_app(model_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = _load_startup_model(model_dir)
        top_k = ModelConfig.from_yaml().shap.top_k
        app.state.service = ScoringService(model, RuleEngine.from_yaml(), top_k=top_k)
        app.state.started = time.monotonic()
        logger.info("loaded model %s", model.version if model else None)
        yield

    app = FastAPI(title="AdShield", version="0.1.0", lifespan=lifespan)

    def to_response(s: ScoredClick) -> ScoreResponse:
        return ScoreResponse(
            risk_score=s.risk_score,
            is_flagged=s.is_flagged,
            top_contributing_features=[
                FeatureContribution(feature=f, value=v, contribution=c) for f, v, c in s.contributions
            ],
            rule_engine_score=s.rule_score,
            rule_engine_fired_rules=s.fired_rules,
            scoring_source=s.source,
            model_version=s.model_version,
            record_id=s.record_id,
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(record: ClickRequest, service: Service) -> ScoreResponse:
        """Score one raw click with the trained model and the rule engine."""
        return to_response(service.score([record])[0])

    @app.post("/score/batch", response_model=BatchScoreResponse)
    def score_batch(
        records: Annotated[list[ClickRequest], Body(min_length=1, max_length=MAX_BATCH)],
        service: Service,
        commit_history: Annotated[
            bool, Query(description="Remember these clicks as IP history. Use false for analysis runs.")
        ] = True,
        top_k: Annotated[
            int | None,
            Query(ge=1, le=20, description="Contributing features per result (default from config)"),
        ] = None,
    ) -> BatchScoreResponse:
        """Score a list of raw clicks; results come back in input order."""
        results = [to_response(s) for s in service.score(records, top_k=top_k, commit_history=commit_history)]
        return BatchScoreResponse(count=len(results), results=results)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, service: Service) -> HealthResponse:
        model = service.model
        return HealthResponse(
            status="ok" if model is not None else "degraded",
            scoring_mode="model" if model is not None else "rules_only",
            model_version=model.version if model else None,
            model_threshold=model.threshold if model else None,
            rules_config_hash=service.engine.config_hash,
            uptime_seconds=round(time.monotonic() - request.app.state.started, 3),
            tracked_ips=len(service.history),
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Validation problems are 422 via FastAPI; anything reaching here is a server fault.
        # Details go to the log, not to the caller.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal scoring error"})

    return app


app = create_app()
