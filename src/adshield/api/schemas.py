"""Pydantic request and response models for the scoring API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

INT32_MAX = 2_147_483_647
MAX_BATCH = 10_000  # records per /score/batch request
CLICK_TIME_MIN = datetime(2000, 1, 1)
CLICK_TIME_MAX = datetime(2100, 1, 1)

# Codes are non-negative 32-bit ints. They are deliberately NOT limited to the
# ranges seen in the training file: a new app/channel id is legitimate traffic,
# and the risk encoder scores unseen combinations at the training base rate.
Code = Annotated[int, Field(strict=True, ge=0, le=INT32_MAX)]


class ClickRequest(BaseModel):
    """One raw click/lead event, exactly the fields in the source data."""

    model_config = ConfigDict(extra="forbid")

    ip: Code = Field(description="Anonymised IP address code")
    app: Code
    device: Code
    os: Code
    channel: Code
    click_time: datetime = Field(
        description="ISO-8601 timestamp. Timezone-aware values are converted to UTC."
    )
    record_id: str | None = Field(
        default=None, max_length=128, description="Optional caller reference, echoed in the response"
    )

    @field_validator("click_time")
    @classmethod
    def _normalise_and_check_time(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            v = v.astimezone(UTC).replace(tzinfo=None)
        if not CLICK_TIME_MIN <= v < CLICK_TIME_MAX:
            raise ValueError(
                f"click_time must be between {CLICK_TIME_MIN.date()} and {CLICK_TIME_MAX.date()}"
            )
        return v


class FeatureContribution(BaseModel):
    feature: str
    value: float | None = Field(
        description="The feature's value for this click; null when not yet measurable (e.g. no click history)"
    )
    contribution: float = Field(
        description="SHAP value in log-odds: positive pushes toward flagging, negative away"
    )


class ScoreResponse(BaseModel):
    risk_score: float = Field(
        description="Model fraud-proxy probability in [0, 1] (rule score / 100 when the rules fallback "
        "was used). Class weighting means this is a ranking score, not a calibrated frequency."
    )
    is_flagged: bool
    top_contributing_features: list[FeatureContribution] = Field(
        description="Largest SHAP contributors for this click; empty when scored by the rules fallback"
    )
    rule_engine_score: float = Field(ge=0, le=100, description="Deterministic rule-baseline score, 0-100")
    rule_engine_fired_rules: list[str] = Field(description="Names of the rules behind rule_engine_score")
    scoring_source: Literal["model", "rules_fallback"]
    model_version: str | None
    record_id: str | None = None


class BatchScoreResponse(BaseModel):
    count: int
    results: list[ScoreResponse] = Field(description="One result per input record, in input order")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    scoring_mode: Literal["model", "rules_only"]
    model_version: str | None
    model_threshold: float | None
    rules_config_hash: str
    uptime_seconds: float
    tracked_ips: int
