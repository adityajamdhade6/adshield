"""Scoring service: per-IP click history, feature pipeline, model, rule engine.

The trailing-window, uniqueness and regularity features describe an IP's
recent behaviour, so a single raw click cannot be featurised in isolation.
The service therefore keeps a bounded in-memory history of recently scored
clicks per IP and featurises each new click against it.

Training/serving skew to be aware of: in training, the cumulative features
(distinct apps, click count, gap std) accumulate over the whole log; here they
accumulate over the retained history only (``retention`` / ``max_per_ip``), and
history is lost on restart. The 10-minute / 1-hour / 24-hour windows are
unaffected as long as retention is at least 24 hours.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from adshield.api.schemas import ClickRequest
from adshield.features.engineering import build_features
from adshield.models.classifier import FraudScorer, TrainedModel
from adshield.models.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

RAW_COLUMNS = ["ip", "app", "device", "os", "channel", "click_time"]
RISK_FEATURE = "cat_risk_app_channel_os"


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed dtypes, so an empty history frame concatenates without turning columns into object."""
    return df.astype({**dict.fromkeys(RAW_COLUMNS[:-1], "int64"), "click_time": "datetime64[ns]"})


class ClickHistory:
    """Bounded per-IP memory of recent clicks, with LRU eviction of whole IPs."""

    def __init__(
        self, retention: timedelta = timedelta(hours=24), max_per_ip: int = 1000, max_ips: int = 500_000
    ):
        self.retention, self.max_per_ip, self.max_ips = retention, max_per_ip, max_ips
        self._rows: OrderedDict[int, list[tuple[int, int, int, int, datetime]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._rows)

    def frame_for(self, ips: set[int]) -> pd.DataFrame:
        rows = [(ip, *r) for ip in ips for r in self._rows.get(ip, ())]
        return _typed(pd.DataFrame(rows, columns=RAW_COLUMNS))

    def add(self, records: list[ClickRequest]) -> None:
        for r in records:
            self._rows.setdefault(r.ip, []).append((r.app, r.device, r.os, r.channel, r.click_time))
            self._rows.move_to_end(r.ip)
        for ip in {r.ip for r in records}:
            rows = self._rows[ip]
            newest = max(t for *_, t in rows)
            kept = [x for x in rows if x[-1] > newest - self.retention]
            self._rows[ip] = kept[-self.max_per_ip :]
        while len(self._rows) > self.max_ips:
            self._rows.popitem(last=False)


@dataclass(frozen=True)
class ScoredClick:
    risk_score: float
    is_flagged: bool
    contributions: list[tuple[str, float | None, float]]  # (feature, value, shap)
    rule_score: float
    fired_rules: list[str]
    source: str
    model_version: str | None
    record_id: str | None


class ScoringService:
    """Thread-safe scorer shared by all requests; models and rules are loaded once."""

    def __init__(
        self,
        model: TrainedModel | None,
        engine: RuleEngine,
        top_k: int = 5,
        history: ClickHistory | None = None,
    ) -> None:
        self.model, self.engine, self.top_k = model, engine, top_k
        self.scorer = FraudScorer(model, engine)
        self.history = history or ClickHistory()
        self._lock = threading.Lock()

    def _features(self, combined: pd.DataFrame) -> pd.DataFrame:
        if self.model is not None:
            return self.model.pipeline.transform(combined)
        # Rules-only mode: no fitted risk encoder, so the traffic-source rule has no
        # evidence (NaN never fires a rule).
        feats = build_features(combined)
        feats[RISK_FEATURE] = np.nan
        return feats

    def score(
        self, records: list[ClickRequest], *, top_k: int | None = None, commit_history: bool = True
    ) -> list[ScoredClick]:
        """Score records in input order. Each sees the stored history plus earlier
        records of the same call (features only look backwards in time). With
        ``commit_history`` they are then added to the history; pass False for
        analysis runs (e.g. scoring an uploaded file) that must not alter live state."""
        batch = _typed(
            pd.DataFrame([{c: getattr(r, c) for c in RAW_COLUMNS} for r in records], columns=RAW_COLUMNS)
        )
        batch["_pos"] = np.arange(len(records))
        with self._lock:
            hist = self.history.frame_for(set(batch["ip"]))
            hist["_pos"] = -1
            combined = pd.concat([hist, batch], ignore_index=True)
            feats = self._features(combined)
            is_batch = (combined["_pos"] >= 0).to_numpy()
            batch_feats = feats[is_batch].set_axis(combined["_pos"][is_batch].to_numpy())
            batch_feats = batch_feats.sort_index()

            scored = self.scorer.score(batch_feats)
            contrib: dict[int, list[tuple[str, float | None, float]]] = {}
            from_model = scored.index[scored["source"] == "model"]
            if self.model is not None and len(from_model):
                try:
                    ex = self.model.explain(batch_feats.loc[from_model], top_k or self.top_k)
                    for r, f, v, c in zip(
                        ex["row"].to_numpy(), ex["feature"], ex["value"], ex["shap"], strict=True
                    ):
                        contrib.setdefault(int(r), []).append(
                            (str(f), None if math.isnan(float(v)) else float(v), float(c))
                        )
                except Exception:  # explanations must never block a score
                    logger.exception("SHAP explanation failed; returning scores without contributions")
            if commit_history:
                self.history.add(records)

        sources = scored["source"].tolist()
        probs = scored["model_probability"].to_numpy(dtype="float64")
        flags = scored["flagged"].to_numpy(dtype=bool)
        rule_scores = scored["rule_score"].to_numpy(dtype="float64")
        fired = scored["fired_rules"].tolist()
        out = []
        for i, rec in enumerate(records):
            rule_score = float(rule_scores[i])
            out.append(
                ScoredClick(
                    risk_score=float(probs[i]) if sources[i] == "model" else rule_score / 100.0,
                    is_flagged=bool(flags[i]),
                    contributions=contrib.get(i, []),
                    rule_score=rule_score,
                    fired_rules=list(fired[i]),
                    source=str(sources[i]),
                    model_version=self.model.version if self.model else None,
                    record_id=rec.record_id,
                )
            )
        return out
