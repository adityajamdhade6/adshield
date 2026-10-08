"""Evaluate fraud scorers on labelled clicks and choose decision thresholds.

Positive class = "fraud" = a click that never converted (``is_attributed == 0``),
the practical fraud-risk proxy used throughout this project. Metrics are
computed with plain NumPy so they are fully inspectable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from adshield.data.loader import load_clicks
from adshield.features.engineering import (
    apply_risk_encoder,
    build_features,
    fit_risk_encoder,
    out_of_fold_risk,
)
from adshield.models.rule_engine import RuleEngine

RISK_KEYS = ("app", "channel", "os")


@dataclass(frozen=True)
class BinaryMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float

    @property
    def confusion_matrix(self) -> pd.DataFrame:
        """Rows = actual, columns = predicted."""
        return pd.DataFrame(
            [[self.tn, self.fp], [self.fn, self.tp]],
            index=["actual_legit", "actual_fraud"],
            columns=["pred_legit", "pred_fraud"],
        )

    def __str__(self) -> str:
        return (
            f"precision={self.precision:.4f} recall={self.recall:.4f} f1={self.f1:.4f}\n"
            f"{self.confusion_matrix.to_string()}"
        )


def binary_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> BinaryMetrics:
    """Precision, recall, F1 and confusion counts. Undefined ratios (0/0) are reported as 0.0."""
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    tn = int((~t & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(tp, fp, fn, tn, precision, recall, f1)


def threshold_sweep(scores: pd.Series, y_true: pd.Series, thresholds: list[float]) -> pd.DataFrame:
    """Precision/recall/F1 and flag rate for each ``score >= threshold`` cut."""
    rows = []
    for th in thresholds:
        m = binary_metrics(y_true, scores >= th)
        rows.append(
            {
                "threshold": th,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "flag_rate": float((scores >= th).mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BaselineReport:
    threshold: float
    cutoff: pd.Timestamp
    n_train: int
    n_test: int
    test_fraud_rate: float
    rule_metrics: BinaryMetrics
    flag_all_metrics: BinaryMetrics
    train_sweep: pd.DataFrame
    rule_fire_rate: pd.Series

    def __str__(self) -> str:
        return "\n".join(
            [
                f"time split at {self.cutoff}: train={self.n_train:,} test={self.n_test:,}",
                f"test fraud-proxy rate: {self.test_fraud_rate:.4%}",
                f"\nRule engine on test, threshold >= {self.threshold:g}:",
                str(self.rule_metrics),
                "\nReference: flag every click as fraud:",
                str(self.flag_all_metrics),
                "\nThreshold sweep on TRAIN (use this to choose the threshold):",
                self.train_sweep.to_string(index=False),
                "\nShare of test clicks each rule fired on:",
                self.rule_fire_rate.to_string(),
            ]
        )


def evaluate_rule_baseline(
    clicks: pd.DataFrame,
    engine: RuleEngine,
    train_frac: float = 0.7,
    sweep_thresholds: list[float] | None = None,
) -> BaselineReport:
    """Score a labelled click log with the rule engine using a forward-in-time split.

    Point-in-time features are computed on the whole log (they only look
    backwards). The categorical risk encoder is fitted on the earlier
    ``train_frac`` of clicks only: train rows get out-of-fold values and test
    rows get values from the fitted encoder, so no label leaks into a score.
    The threshold sweep is on the train period; the headline metrics are on
    the later test period.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    clicks = clicks.sort_values("click_time", kind="stable")
    cutoff_pos = int(len(clicks) * train_frac)
    if cutoff_pos == 0 or cutoff_pos == len(clicks):
        raise ValueError("not enough rows for a train/test split")
    cutoff = clicks["click_time"].iloc[cutoff_pos]
    is_train = (clicks["click_time"] < cutoff).to_numpy()
    train, test = clicks[is_train], clicks[~is_train]
    if train.empty or test.empty:
        raise ValueError("time split produced an empty side")

    is_fraud = clicks["is_attributed"] == 0
    features = build_features(clicks)
    encoder = fit_risk_encoder(train, is_fraud[train.index], RISK_KEYS)
    risk = pd.concat(
        [
            out_of_fold_risk(train, is_fraud[train.index], RISK_KEYS),
            apply_risk_encoder(test, encoder),
        ]
    )
    features[encoder.feature_name] = risk.reindex(features.index)

    scores = engine.score_detailed(features)
    threshold = engine.config.decision_threshold
    y_test = is_fraud[test.index]
    test_scores = scores.score[test.index]
    sweep = threshold_sweep(
        scores.score[train.index],
        is_fraud[train.index],
        sweep_thresholds or [10, 20, 30, 40, 50, 60, 70, 80, 90],
    )
    return BaselineReport(
        threshold=threshold,
        cutoff=cutoff,
        n_train=len(train),
        n_test=len(test),
        test_fraud_rate=float(y_test.mean()),
        rule_metrics=binary_metrics(y_test, test_scores >= threshold),
        flag_all_metrics=binary_metrics(y_test, np.ones(len(y_test), dtype=bool)),
        train_sweep=sweep,
        rule_fire_rate=(scores.contributions.loc[test.index] > 0).mean().rename("fire_rate"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the rule-based baseline on labelled clicks.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()
    clicks = load_clicks(args.path, nrows=args.nrows)
    print(evaluate_rule_baseline(clicks, RuleEngine.from_yaml(), args.train_frac))


if __name__ == "__main__":
    main()
