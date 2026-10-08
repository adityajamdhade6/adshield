"""Deterministic, explainable rule-based fraud scorer.

The explainable floor a learned model must beat, and the fallback if that
model misbehaves in production. Rules, thresholds and weights live in
``configs/rules.yaml``; nothing here depends on a machine-learning library.

A click's risk score is the sum of the points of every rule whose conditions
all hold, capped at ``max_score``. Because the rules are named and additive,
every score can be decomposed exactly into the rules that produced it.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adshield.config import load_yaml

_OPS: Final[dict[str, Callable[[np.ndarray, float], np.ndarray]]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
}


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    op: Literal[">=", ">", "<=", "<"]
    threshold: float


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    points: float = Field(gt=0)
    conditions: Annotated[list[Condition], Field(min_length=1)]


class RulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    max_score: float = Field(gt=0)
    decision_threshold: float = Field(ge=0)
    rules: Annotated[list[Rule], Field(min_length=1)]

    @field_validator("rules")
    @classmethod
    def _unique_names(cls, rules: list[Rule]) -> list[Rule]:
        names = [r.name for r in rules]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate rule names: {dupes}")
        return rules

    @model_validator(mode="after")
    def _threshold_reachable(self) -> RulesConfig:
        if self.decision_threshold > self.max_score:
            raise ValueError("decision_threshold cannot exceed max_score")
        return self


@dataclass(frozen=True)
class RuleScores:
    """Scores for a batch of rows, with full per-rule attribution.

    ``contributions[r, rule]`` is the points ``rule`` added to row ``r`` (0 if
    it did not fire). ``raw_points`` is their sum; ``score`` is the sum capped
    at ``max_score``. All frames share the input's index.
    """

    score: pd.Series
    raw_points: pd.Series
    contributions: pd.DataFrame
    config_hash: str

    def fired_rules(self) -> pd.Series:
        """Per row, the tuple of names of rules that fired, highest points first."""
        names = self.contributions.columns.to_numpy()
        values = self.contributions.to_numpy()
        fired = [
            tuple(names[i] for i in sorted(np.flatnonzero(row), key=lambda j: -row[j])) for row in values
        ]
        return pd.Series(fired, index=self.contributions.index, name="fired_rules", dtype=object)


class RuleEngine:
    """Scores rows of engineered features with the rules from a ``RulesConfig``."""

    def __init__(self, config: RulesConfig) -> None:
        self.config = config
        canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
        self.config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, name: str = "rules.yaml", config_dir: Path | None = None) -> RuleEngine:
        return cls(RulesConfig.model_validate(load_yaml(name, config_dir)))

    @property
    def required_features(self) -> list[str]:
        return sorted({c.feature for r in self.config.rules for c in r.conditions})

    def _check_features(self, features: pd.DataFrame) -> None:
        missing = [f for f in self.required_features if f not in features.columns]
        if missing:
            raise ValueError(f"features missing for configured rules: {missing}")

    def _rule_mask(self, features: pd.DataFrame, rule: Rule) -> np.ndarray:
        # Comparisons against NaN are False, so a rule never fires on missing evidence.
        mask = np.ones(len(features), dtype=bool)
        for cond in rule.conditions:
            mask &= _OPS[cond.op](features[cond.feature].to_numpy(dtype="float64"), cond.threshold)
        return mask

    def score_detailed(self, features: pd.DataFrame) -> RuleScores:
        """Score every row and attribute the score to the rules that fired."""
        self._check_features(features)
        contrib = pd.DataFrame(
            {
                rule.name: np.where(self._rule_mask(features, rule), rule.points, 0.0)
                for rule in self.config.rules
            },
            index=features.index,
        )
        raw = contrib.sum(axis=1).rename("raw_points")
        score = raw.clip(upper=self.config.max_score).rename("risk_score")
        return RuleScores(score, raw, contrib, self.config_hash)

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Risk score in [0, max_score] per row."""
        return self.score_detailed(features).score

    def flag(self, features: pd.DataFrame) -> pd.Series:
        """True where the score reaches ``decision_threshold``."""
        return (self.score(features) >= self.config.decision_threshold).rename("flagged")

    def audit_trail(self, features: pd.DataFrame) -> pd.DataFrame:
        """Long-form audit log: one row per (scored row, rule that fired).

        Columns: ``row`` (input index label), ``rule``, ``points``,
        ``description``, ``evidence`` (the feature values that triggered the
        rule), ``risk_score`` (the row's final capped score) and
        ``config_hash`` (identifies the exact rules/weights used). Rows where
        no rule fired have no entries; their score is 0.
        """
        result = self.score_detailed(features)
        records: list[dict[str, object]] = []
        for rule in self.config.rules:
            fired = result.contributions[rule.name].to_numpy() > 0
            for pos in np.flatnonzero(fired):
                evidence = "; ".join(
                    f"{c.feature}={features[c.feature].iloc[pos]:g} ({c.op} {c.threshold:g})"
                    for c in rule.conditions
                )
                records.append(
                    {
                        "row": features.index[int(pos)],
                        "rule": rule.name,
                        "points": rule.points,
                        "description": rule.description,
                        "evidence": evidence,
                        "risk_score": result.score.iloc[pos],
                        "config_hash": self.config_hash,
                    }
                )
        columns = ["row", "rule", "points", "description", "evidence", "risk_score", "config_hash"]
        return pd.DataFrame.from_records(records, columns=columns)
