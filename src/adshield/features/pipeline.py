"""Reproducible feature pipeline shared by training and inference.

Wraps the production feature functions in ``adshield.features.engineering``
together with the one piece of learned state they need: the categorical risk
encoder fitted on the training period.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adshield.features.engineering import (
    RiskEncoder,
    build_features,
    fit_risk_encoder,
    out_of_fold_risk,
)

FEATURE_NAMES: tuple[str, ...] = (
    "clicks_ip_10min",
    "clicks_ip_1h",
    "clicks_ip_24h",
    "ip_click_count",
    "ip_distinct_apps",
    "ip_distinct_channels",
    "ip_distinct_devices",
    "ip_unique_ratio",
    "ip_gap_count",
    "ip_gap_std_s",
)


@dataclass(frozen=True)
class FeaturePipeline:
    """Fitted feature pipeline: a fixed column order plus the fitted risk encoder."""

    encoder: RiskEncoder
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        train_clicks: pd.DataFrame,
        is_fraud: pd.Series,
        keys: tuple[str, ...] = ("app", "channel", "os"),
        smoothing: float = 20.0,
    ) -> FeaturePipeline:
        """Fit the risk encoder on training clicks only."""
        encoder = fit_risk_encoder(train_clicks, is_fraud, keys, smoothing)
        return cls(encoder=encoder, feature_names=(*FEATURE_NAMES, encoder.feature_name))

    def transform(
        self,
        clicks: pd.DataFrame,
        train_labels: pd.Series | None = None,
        n_folds: int = 5,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Feature matrix for ``clicks``, columns in ``feature_names`` order.

        ``clicks`` must be the click log *including the history* needed by the
        trailing-window features (e.g. the full log, then slice the rows you
        want). All history features look backwards only, so rows outside the
        slice you later use cannot leak into it.

        Pass ``train_labels`` (fraud labels indexed like the training rows of
        ``clicks``) when building the training matrix: those rows then get
        out-of-fold risk values instead of values computed from their own
        labels. Leave it ``None`` for validation, test and production rows.
        """
        feats = build_features(clicks, self.encoder)
        if train_labels is not None:
            oof = out_of_fold_risk(
                clicks.loc[train_labels.index],
                train_labels,
                self.encoder.keys,
                self.encoder.smoothing,
                n_folds,
                seed,
            )
            feats.loc[train_labels.index, self.encoder.feature_name] = oof.to_numpy()
        return feats[list(self.feature_names)]
