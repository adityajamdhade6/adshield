"""Pure, testable logic behind the dashboard: parsing uploads, plain-language reasons, tradeoff tables."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from adshield.api.schemas import MAX_BATCH, ClickRequest
from adshield.dashboard.client import ScoringClient

REQUIRED_COLUMNS = ("ip", "app", "device", "os", "channel", "click_time")
ID_COLUMNS = ("lead_id", "record_id", "id")
LABEL_COLUMN = "is_attributed"
CODE_COLUMNS = ("ip", "app", "device", "os", "channel")


class UploadError(ValueError):
    """The uploaded file cannot be scored. The message is written for the person who uploaded it."""


@dataclass(frozen=True)
class RowError:
    row: int  # spreadsheet row number: the header is row 1, the first lead is row 2
    field: str
    message: str

    def __str__(self) -> str:
        return f"Row {self.row}, '{self.field}': {self.message}"


@dataclass(frozen=True)
class ParsedUpload:
    records: list[ClickRequest]
    lead_ids: list[str]
    converted: np.ndarray | None  # True where the click later converted; None if the file has no label


# ------------------------------------------------------------------ reading the upload


def read_upload(data: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise UploadError(f"This does not look like a readable CSV file ({exc.__class__.__name__}).") from exc
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise UploadError(
            f"The file is missing required column(s): {', '.join(missing)}. "
            f"It needs: {', '.join(REQUIRED_COLUMNS)}."
        )
    if df.empty:
        raise UploadError("The file has a header but no rows.")
    if len(df) > MAX_BATCH:
        raise UploadError(
            f"The file has {len(df):,} rows; the limit is {MAX_BATCH:,} per upload. Please split it."
        )
    return df


def parse_leads(df: pd.DataFrame, max_errors: int = 50) -> tuple[ParsedUpload | None, list[RowError]]:
    """Validate every row with the same model the API uses. Returns (parsed, []) or (None, errors)."""
    errors: list[RowError] = []
    records: list[ClickRequest] = []
    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    lead_ids: list[str] = []

    for i, row in enumerate(df.to_dict("records")):
        sheet_row = i + 2
        raw: dict[str, Any] = {}
        row_errors: list[RowError] = []
        for col in REQUIRED_COLUMNS:
            val = str(row[col]).strip()
            if val == "":
                row_errors.append(RowError(sheet_row, col, "is empty"))
            elif col in CODE_COLUMNS:
                try:
                    raw[col] = int(val)
                except ValueError:
                    row_errors.append(RowError(sheet_row, col, f"must be a whole number, got '{val}'"))
            else:
                raw[col] = val
        if not row_errors:
            try:
                records.append(ClickRequest.model_validate(raw))
            except ValidationError as exc:
                for e in exc.errors():
                    row_errors.append(
                        RowError(sheet_row, str(e["loc"][0]), e["msg"].removeprefix("Value error, "))
                    )
        errors.extend(row_errors)
        lead_ids.append(str(row[id_col]).strip() if id_col and str(row[id_col]).strip() else str(i + 1))
        if len(errors) >= max_errors:
            break
    if errors:
        return None, errors

    converted = None
    if LABEL_COLUMN in df.columns:
        labels = df[LABEL_COLUMN].str.strip()
        if labels.isin(["0", "1"]).all():
            converted = (labels == "1").to_numpy()
    return ParsedUpload(records, lead_ids, converted), []


# ------------------------------------------------------------------ scoring


def score_upload(parsed: ParsedUpload, client: ScoringClient) -> pd.DataFrame:
    """Send the leads to the API in time order (its features depend on earlier clicks) and
    return results in the original file order."""
    times = np.array([r.click_time for r in parsed.records], dtype="datetime64[ns]")
    order = np.argsort(times, kind="stable")
    api_results = client.score_batch([parsed.records[i] for i in order])
    by_original: list[dict[str, Any]] = [{}] * len(order)
    for pos, res in zip(order, api_results, strict=True):
        by_original[pos] = res

    df = pd.DataFrame(
        {
            "lead_id": parsed.lead_ids,
            "risk_score": [r["risk_score"] for r in by_original],
            "model_flagged": [r["is_flagged"] for r in by_original],
            "rule_score": [r["rule_engine_score"] for r in by_original],
            "fired_rules": [r["rule_engine_fired_rules"] for r in by_original],
            "source": [r["scoring_source"] for r in by_original],
            "contributions": [r["top_contributing_features"] for r in by_original],
        }
    )
    if parsed.converted is not None:
        df["converted"] = parsed.converted
    return df


# ------------------------------------------------------------------ plain-language explanations

# feature -> sentence template using the value, written for a non-technical reader.
_TEMPLATES: dict[str, str] = {
    "clicks_ip_10min": "{v:.0f} click(s) from this IP in the last 10 minutes",
    "clicks_ip_1h": "{v:.0f} click(s) from this IP in the last hour",
    "clicks_ip_24h": "{v:.0f} click(s) from this IP in the last 24 hours",
    "ip_click_count": "this IP has made {v:.0f} click(s) in total so far",
    "ip_distinct_apps": "this IP has clicked {v:.0f} different app(s)",
    "ip_distinct_channels": "this IP has come through {v:.0f} different channel(s)",
    "ip_distinct_devices": "this IP has used {v:.0f} different device type(s)",
    "ip_unique_ratio": "only {pct:.0f}% of this IP's clicks were different from each other "
    "(low means the same click is repeated)",
    "ip_gap_count": "{v:.0f} time gap(s) between this IP's clicks measured so far",
    "ip_gap_std_s": "the time between this IP's clicks varies by only {v:.1f} seconds "
    "(near 0 means clockwork-like spacing)",
    "cat_risk_app_channel_os": "this app / channel / device-OS combination has a {pct:.1f}% "
    "history of clicks that never convert",
}
_NO_DATA = {
    "ip_gap_std_s": "not enough clicks from this IP yet to judge how evenly spaced they are",
}


def describe_feature(feature: str, value: float | None) -> str:
    """One plain-English clause describing a feature's value for a click."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _NO_DATA.get(feature, f"{feature.replace('_', ' ')} is not available yet")
    template = _TEMPLATES.get(feature)
    if template is None:
        return f"{feature.replace('_', ' ')} = {value:g}"
    return template.format(v=value, pct=100 * value)


def reason_text(contribution: dict[str, Any]) -> str:
    arrow = "▲" if contribution["contribution"] > 0 else "▼"
    return f"{arrow} {describe_feature(contribution['feature'], contribution['value'])}"


def top_reasons(contributions: list[dict[str, Any]], n: int = 2) -> str:
    """The n strongest drivers (▲ raises risk, ▼ lowers it) as one line of text."""
    ranked = sorted(contributions, key=lambda c: abs(c["contribution"]), reverse=True)[:n]
    return "; ".join(reason_text(c) for c in ranked) if ranked else "Scored by rules only (no model reasons)"


def results_table(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The manager-facing table at the chosen threshold, highest risk first."""
    out = pd.DataFrame(
        {
            "Lead ID": results["lead_id"],
            "Risk score": results["risk_score"].round(3),
            "Flagged": np.where(results["risk_score"] >= threshold, "Yes", "No"),
            "Why (top 2 reasons)": results["contributions"].map(top_reasons),
        }
    )
    return out.sort_values("Risk score", ascending=False, kind="stable")


# ------------------------------------------------------------------ threshold tradeoff


def tradeoff_curve(results: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Flag counts (and, when outcomes are known, their cost and benefit) at each threshold."""
    scores = results["risk_score"].to_numpy()
    rows = []
    for t in thresholds:
        flagged = scores >= t
        row: dict[str, float] = {
            "threshold": float(t),
            "flagged": int(flagged.sum()),
            "pct_flagged": 100 * flagged.mean(),
        }
        if "converted" in results:
            conv = results["converted"].to_numpy()
            n_conv, n_non = conv.sum(), (~conv).sum()
            row["pct_nonconverting_flagged"] = (
                100 * (flagged & ~conv).sum() / n_non if n_non else float("nan")
            )
            row["pct_conversions_flagged"] = 100 * (flagged & conv).sum() / n_conv if n_conv else float("nan")
            row["conversions_flagged"] = int((flagged & conv).sum())
        rows.append(row)
    return pd.DataFrame(rows)
