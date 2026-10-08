"""Trained fraud classifier layered on top of the rule-based baseline.

Pipeline: chronological split -> feature pipeline (risk encoder fitted on the
training period) -> XGBoost with class-imbalance weighting -> explicit grid
search with every trial logged -> evaluation on a held-out later period ->
SHAP explanations -> versioned artifacts on disk.

The rule engine is not replaced. ``FraudScorer`` runs the model and falls back
to the rules whenever the model is unavailable or produces unusable output,
and always attaches the rule engine's fired rules to its result.

Label: positive = fraud proxy = ``is_attributed == 0`` (click never converted).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import average_precision_score, roc_auc_score

from adshield.config import load_yaml
from adshield.data.loader import load_clicks
from adshield.features.pipeline import FeaturePipeline
from adshield.models.comparison import write_report
from adshield.models.evaluate import binary_metrics
from adshield.models.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports"


# --------------------------------------------------------------------------- config


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_frac: float = Field(gt=0, lt=1)
    val_frac: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _leaves_test(self) -> SplitConfig:
        if self.train_frac + self.val_frac >= 1:
            raise ValueError("train_frac + val_frac must be < 1 so a test period remains")
        return self


class RiskEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keys: list[str] = Field(min_length=1)
    smoothing: float = Field(ge=0)
    n_folds: int = Field(ge=2)


class XGBoostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixed: dict[str, Any]
    grid: dict[str, list[Any]]

    @model_validator(mode="after")
    def _check(self) -> XGBoostConfig:
        if not self.grid or any(len(v) == 0 for v in self.grid.values()):
            raise ValueError("grid must be non-empty and every parameter needs at least one value")
        if overlap := set(self.grid) & set(self.fixed):
            raise ValueError(f"parameters both fixed and in grid: {sorted(overlap)}")
        return self


class ShapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_size: int = Field(gt=0)
    top_k: int = Field(gt=0)


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_margin: float = Field(gt=0)
    min_minority_rows: int = Field(ge=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    split: SplitConfig
    risk_encoder: RiskEncoderConfig
    xgboost: XGBoostConfig
    shap: ShapConfig
    comparison: ComparisonConfig

    @classmethod
    def from_yaml(cls, name: str = "model.yaml", config_dir: Path | None = None) -> ModelConfig:
        return cls.model_validate(load_yaml(name, config_dir))


# --------------------------------------------------------------------------- time split


@dataclass(frozen=True)
class TimeSplit:
    """Row labels for each period plus the boundary timestamps."""

    train: pd.Index
    val: pd.Index
    test: pd.Index
    val_start: pd.Timestamp
    test_start: pd.Timestamp


def time_split(clicks: pd.DataFrame, train_frac: float, val_frac: float) -> TimeSplit:
    """Split by ``click_time`` into consecutive train / validation / test periods.

    Rows are never shuffled. Boundaries are timestamps, so clicks sharing a
    timestamp always land in the same period, and every validation click is at
    or after every training click (likewise test after validation). A random
    split would let the model see clicks from after the ones it is scored on,
    and, because features such as per-IP counts are autocorrelated in time,
    inflate every metric.

    Note: a click's label (did it later convert?) is only known some time after
    the click, so training labels near the boundary are slightly "from the
    future" relative to the first validation clicks. We do not purge that
    window; with conversions arriving within hours it affects a thin sliver.
    """
    if not clicks.index.is_unique:
        raise ValueError("clicks must have a unique index")
    n = len(clicks)
    if n < 10:
        raise ValueError("need at least 10 clicks to split")
    times = clicks["click_time"]
    ordered = times.sort_values(kind="stable").to_numpy()
    val_start = pd.Timestamp(ordered[int(n * train_frac)])
    test_start = pd.Timestamp(ordered[int(n * (train_frac + val_frac))])
    train_mask = times < val_start
    val_mask = (times >= val_start) & (times < test_start)
    test_mask = times >= test_start
    split = TimeSplit(
        clicks.index[train_mask.to_numpy()],
        clicks.index[val_mask.to_numpy()],
        clicks.index[test_mask.to_numpy()],
        val_start,
        test_start,
    )
    if min(len(split.train), len(split.val), len(split.test)) == 0:
        raise ValueError("time split produced an empty period (too many identical timestamps?)")
    return split


# --------------------------------------------------------------------------- metrics


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Threshold-free and thresholded metrics for a fraud score (higher = riskier).

    PR-AUC vs ROC-AUC: ROC-AUC's false-positive rate divides by the (huge)
    number of negatives, so under heavy imbalance a scorer can raise thousands
    of false alarms and still show a tiny FPR and a high ROC-AUC. PR-AUC uses
    precision (alarms that are real) and recall, ignores true negatives, and
    has an honest floor equal to the class prevalence, so it shows what the
    minority class actually costs. It is therefore the headline metric.

    With the proxy label the fraud class may be the *majority*; then the
    informative PR-AUC is that of the minority class, reported as
    ``pr_auc_minority``. ``pr_auc`` is always for the fraud-proxy class.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype="float64")
    if y.all() or not y.any():
        raise ValueError("evaluation set must contain both classes")
    pred = s >= threshold
    prevalence = float(y.mean())
    minority_is_target = prevalence <= 0.5
    y_min = y if minority_is_target else ~y
    s_min = s if minority_is_target else -s
    pred_min = pred if minority_is_target else ~pred

    def view(t: np.ndarray, p: np.ndarray) -> dict[str, float]:
        m = binary_metrics(t, p)
        return {"precision": m.precision, "recall": m.recall, "f1": m.f1}

    m = binary_metrics(y, pred)
    return {
        "n_rows": int(len(y)),
        "prevalence": prevalence,
        "minority_is_target": minority_is_target,
        "minority_prevalence": float(y_min.mean()),
        "threshold": float(threshold),
        "target_view": view(y, pred),
        "minority_view": view(y_min, pred_min),
        "confusion": {"tn": m.tn, "fp": m.fp, "fn": m.fn, "tp": m.tp},
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "pr_auc_baseline": prevalence,
        "pr_auc_minority": float(average_precision_score(y_min, s_min)),
    }


def tune_threshold(y_true: np.ndarray, scores: np.ndarray, max_candidates: int = 2000) -> float:
    """Score threshold (flag when ``score >= threshold``) maximising minority-class F1.

    The minority class is used because with the proxy label the fraud class can
    be the majority (most clicks never convert); F1 for that class is then
    maximised by flagging everything, which is a useless operating point. When
    fraud really is the rare class this is simply the fraud-class F1 optimum.
    Candidates are the distinct scores (thinned to ``max_candidates`` quantiles).
    Use validation data, never the test set.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype="float64")
    if y.all() or not y.any():
        raise ValueError("threshold tuning needs both classes")
    cand = np.unique(s)
    if len(cand) > max_candidates:
        cand = np.unique(np.quantile(cand, np.linspace(0, 1, max_candidates)))
    pos, neg = np.sort(s[y]), np.sort(s[~y])
    fn = np.searchsorted(pos, cand, side="left")  # target positives scored below threshold
    tp = len(pos) - fn
    tn = np.searchsorted(neg, cand, side="left")  # target negatives scored below threshold
    fp = len(neg) - tn
    if y.mean() > 0.5:  # minority class = target negatives
        tp, fp, fn = tn, fn, fp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(cand)), where=denom > 0)
    return float(cand[int(np.argmax(f1))])


# --------------------------------------------------------------------------- model


@dataclass
class TrainedModel:
    """Fitted booster plus everything needed to featurise, score and explain a click."""

    version: str
    booster: xgb.XGBClassifier
    pipeline: FeaturePipeline
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)
    _explainer: shap.TreeExplainer | None = field(default=None, repr=False, compare=False)

    @property
    def feature_names(self) -> list[str]:
        return list(self.pipeline.feature_names)

    def _matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.feature_names if f not in features.columns]
        if missing:
            raise ValueError(f"features missing for model: {missing}")
        return features[self.feature_names]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of the fraud-proxy class. Class weighting shifts these from true
        frequencies, so treat them as scores and use ``threshold``, not 0.5."""
        return self.booster.predict_proba(self._matrix(features))[:, 1]

    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer

    def shap_values(self, features: pd.DataFrame) -> np.ndarray:
        """Per-row, per-feature SHAP values in log-odds units (rows x features)."""
        return np.asarray(self.explainer().shap_values(self._matrix(features)))

    def explain(self, features: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Long-form top contributors per row, for the API's explainability output.

        Columns: ``row`` (input index label), ``rank`` (1 = largest |SHAP|),
        ``feature``, ``value`` (the feature's value), ``shap`` (log-odds
        contribution) and ``direction`` ("raises_risk" / "lowers_risk").
        SHAP values plus the explainer base value sum to the model's raw score.
        """
        matrix = self._matrix(features)
        values = self.shap_values(features)
        k = min(top_k, values.shape[1])
        top = np.argsort(-np.abs(values), axis=1, kind="stable")[:, :k]
        rows = []
        for i in range(len(matrix)):
            for rank, j in enumerate(top[i], start=1):
                s = float(values[i, j])
                rows.append(
                    {
                        "row": matrix.index[i],
                        "rank": rank,
                        "feature": self.feature_names[j],
                        "value": matrix.iloc[i, j],
                        "shap": s,
                        "direction": "raises_risk" if s > 0 else "lowers_risk",
                    }
                )
        return pd.DataFrame(rows, columns=["row", "rank", "feature", "value", "shap", "direction"])

    def global_importance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature over ``features``, largest first."""
        imp = np.abs(self.shap_values(features)).mean(axis=0)
        out = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": imp})
        return out.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- fallback layer


class FraudScorer:
    """Model-first scorer that falls back to the rule engine and always reports the rules.

    The model's probability decides the flag when it is healthy. The rule
    engine's score and fired rules are attached to every row, and replace the
    model entirely, per row or per batch, when the model is missing, raises,
    is missing input features, or returns non-finite / out-of-range output.
    """

    def __init__(self, model: TrainedModel | None, engine: RuleEngine) -> None:
        self.model = model
        self.engine = engine

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        rules = self.engine.score_detailed(features)
        rule_flag = (rules.score >= self.engine.config.decision_threshold).to_numpy()
        n = len(features)
        probs = np.full(n, np.nan)
        batch_reason = None
        if self.model is None:
            batch_reason = "no_model"
        else:
            try:
                probs = np.asarray(self.model.predict_proba(features), dtype="float64")
            except Exception as exc:  # any model failure must degrade to rules, not crash scoring
                logger.error("model scoring failed, falling back to rules: %s", exc)
                batch_reason = "model_error"
        usable = np.isfinite(probs) & (probs >= 0) & (probs <= 1)
        if batch_reason is not None:
            usable[:] = False
        model_flag = probs >= self.model.threshold if self.model is not None else rule_flag
        reason = np.where(usable, "", batch_reason or "invalid_model_output")
        return pd.DataFrame(
            {
                "flagged": np.where(usable, model_flag, rule_flag),
                "source": np.where(usable, "model", "rules_fallback"),
                "fallback_reason": reason,
                "model_probability": np.where(usable, probs, np.nan),
                "rule_score": rules.score.to_numpy(),
                "fired_rules": rules.fired_rules().to_numpy(),
            },
            index=features.index,
        )


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainRun:
    model: TrainedModel
    metrics: dict[str, Any]
    trials: list[dict[str, Any]]
    shap_importance: pd.DataFrame
    directory: Path


def scale_pos_weight(y_train: np.ndarray) -> float:
    """Negatives / positives in the training period (XGBoost's imbalance weighting).

    Upweights the positive (fraud-proxy) class so the loss does not ignore a
    rare class. If positives are the majority this is below 1 and downweights
    them, i.e. it balances the classes whichever way the skew runs.
    """
    y = np.asarray(y_train).astype(bool)
    if y.all() or not y.any():
        raise ValueError("training period must contain both classes")
    return float((~y).sum()) / float(y.sum())


def grid_trials(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination in the grid, in a deterministic order."""
    keys = sorted(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))]


def run_grid_search(
    config: ModelConfig,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    spw: float,
) -> tuple[xgb.XGBClassifier, list[dict[str, Any]], int]:
    """Train one model per grid point; log every trial; return the best by validation PR-AUC.

    Selection metric: validation ``pr_auc_minority`` (see ``evaluate_scores``).
    Early stopping on validation logloss picks the number of trees per trial.
    The test period is never touched here.
    """
    trials: list[dict[str, Any]] = []
    best_model: xgb.XGBClassifier | None = None
    best_idx = -1
    combos = grid_trials(config.xgboost.grid)
    for i, params in enumerate(combos):
        start = time.perf_counter()
        model = xgb.XGBClassifier(
            **config.xgboost.fixed,
            **params,
            scale_pos_weight=spw,
            random_state=config.seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        val_probs = model.predict_proba(x_val)[:, 1]
        val_metrics = evaluate_scores(y_val, val_probs, 0.5)
        trial = {
            "trial": i,
            "params": params,
            "best_iteration": int(model.best_iteration),
            "val_logloss": float(model.best_score),
            "val_pr_auc_minority": val_metrics["pr_auc_minority"],
            "val_roc_auc": val_metrics["roc_auc"],
            "seconds": round(time.perf_counter() - start, 2),
        }
        trials.append(trial)
        logger.info("trial %d/%d params=%s -> %s", i + 1, len(combos), params, trial)
        if best_idx < 0 or trial["val_pr_auc_minority"] > trials[best_idx]["val_pr_auc_minority"]:
            best_model, best_idx = model, i
    assert best_model is not None
    for t in trials:
        t["selected"] = t["trial"] == best_idx
    logger.info("selected trial %d: %s", best_idx, trials[best_idx])
    return best_model, trials, best_idx


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_and_evaluate(
    clicks: pd.DataFrame,
    config: ModelConfig,
    engine: RuleEngine,
    models_dir: Path = MODELS_DIR,
    version: str | None = None,
    data_source: str = "<in-memory>",
) -> TrainRun:
    """Train, evaluate on the held-out test period, compare to the rules, and save everything."""
    version = version or datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    run_dir = models_dir / version
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; choose a new version tag")

    split = time_split(clicks, config.split.train_frac, config.split.val_frac)
    y_all = (clicks["is_attributed"] == 0).rename("is_fraud")
    y_train, y_val, y_test = (y_all[idx].to_numpy() for idx in (split.train, split.val, split.test))

    enc = config.risk_encoder
    pipeline = FeaturePipeline.fit(
        clicks.loc[split.train], y_all[split.train], tuple(enc.keys), enc.smoothing
    )
    features = pipeline.transform(
        clicks, train_labels=y_all[split.train], n_folds=enc.n_folds, seed=config.seed
    )
    x_train, x_val, x_test = (features.loc[idx] for idx in (split.train, split.val, split.test))

    spw = scale_pos_weight(y_train)
    logger.info("scale_pos_weight=%.5f (train fraud-proxy prevalence %.4f%%)", spw, 100 * y_train.mean())
    booster, trials, best_idx = run_grid_search(config, x_train, y_train, x_val, y_val, spw)

    val_probs = booster.predict_proba(x_val)[:, 1]
    threshold = tune_threshold(y_val, val_probs)
    model = TrainedModel(version, booster, pipeline, threshold)

    test_probs = model.predict_proba(x_test)
    # Give the rule baseline the same tuning budget as the model: a threshold
    # chosen on validation. Its configured threshold is reported as well.
    rule_val_scores = engine.score(x_val).to_numpy()
    rule_tuned_threshold = tune_threshold(y_val, rule_val_scores)
    rule_result = engine.score_detailed(x_test)
    test_metrics = {
        "n_rows": int(len(x_test)),
        "prevalence": float(y_test.mean()),
        "model": evaluate_scores(y_test, test_probs, threshold),
        "rules": evaluate_scores(y_test, rule_result.score.to_numpy(), engine.config.decision_threshold),
        "rules_tuned": evaluate_scores(y_test, rule_result.score.to_numpy(), rule_tuned_threshold),
        "flag_all": evaluate_scores(y_test, np.ones(len(y_test)), 1.0),  # constant score
        "rules_threshold": engine.config.decision_threshold,
        "rules_tuned_threshold": rule_tuned_threshold,
        "rules_config_hash": engine.config_hash,
    }

    rng = np.random.default_rng(config.seed)
    n_shap = min(config.shap.sample_size, len(x_test))
    sample = x_test.iloc[np.sort(rng.choice(len(x_test), n_shap, replace=False))]
    importance = model.global_importance(sample)

    metrics: dict[str, Any] = {
        "version": version,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data_source": data_source,
        "data": {
            "n_rows": int(len(clicks)),
            "train_rows": int(len(split.train)),
            "val_rows": int(len(split.val)),
            "test_rows": int(len(split.test)),
            "val_start": str(split.val_start),
            "test_start": str(split.test_start),
            "train_prevalence": float(y_train.mean()),
            "val_prevalence": float(y_val.mean()),
            "test_prevalence": float(y_test.mean()),
        },
        "feature_names": model.feature_names,
        "scale_pos_weight": spw,
        "selected_trial": best_idx,
        "selected_params": trials[best_idx]["params"],
        "best_iteration": trials[best_idx]["best_iteration"],
        "threshold": threshold,
        "threshold_rule": "maximises minority-class F1 on the validation period",
        "val": evaluate_scores(y_val, val_probs, threshold),
        "test": test_metrics,
        "comparison_min_margin": config.comparison.min_margin,
        "comparison_min_minority_rows": config.comparison.min_minority_rows,
        "shap_top_features": importance.head(10).to_dict("records"),
        "libraries": {
            "xgboost": xgb.__version__,
            "shap": shap.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "seed": config.seed,
    }

    run_dir.mkdir(parents=True)
    booster.save_model(run_dir / "model.json")
    joblib.dump(
        {"pipeline": pipeline, "threshold": threshold, "version": version}, run_dir / "pipeline.joblib"
    )
    (run_dir / "trials.json").write_text(json.dumps(trials, indent=2))
    importance.to_csv(run_dir / "shap_importance.csv", index=False)
    metrics["artifacts"] = {
        name: _sha256(run_dir / name)
        for name in ("model.json", "pipeline.joblib", "trials.json", "shap_importance.csv")
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("saved run to %s", run_dir)
    return TrainRun(model, metrics, trials, importance, run_dir)


def load_model(run_dir: str | Path) -> TrainedModel:
    """Load a saved run, verifying artifact checksums recorded in metrics.json first."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for name, expected in metrics["artifacts"].items():
        actual = _sha256(run_dir / name)
        if actual != expected:
            raise ValueError(f"{run_dir / name} checksum mismatch: artifact changed since training")
    saved = joblib.load(run_dir / "pipeline.joblib")
    booster = xgb.XGBClassifier()
    booster.load_model(run_dir / "model.json")
    return TrainedModel(saved["version"], booster, saved["pipeline"], saved["threshold"], metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the fraud classifier.")
    parser.add_argument("path", type=Path, help="raw click CSV")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--version", default=None, help="version tag (default: UTC timestamp)")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "model_comparison.md")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    clicks = load_clicks(args.path, nrows=args.nrows)
    run = train_and_evaluate(
        clicks,
        ModelConfig.from_yaml(),
        RuleEngine.from_yaml(),
        models_dir=args.models_dir,
        version=args.version,
        data_source=str(args.path),
    )
    write_report(run.metrics, args.report)
    print(f"model: {run.directory}\nreport: {args.report}")


if __name__ == "__main__":
    main()
