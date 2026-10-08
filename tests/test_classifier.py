"""Tests for the trained model layer, on synthetic labelled traffic."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from adshield.models.classifier import (
    FraudScorer,
    ModelConfig,
    evaluate_scores,
    grid_trials,
    load_model,
    scale_pos_weight,
    time_split,
    train_and_evaluate,
    tune_threshold,
)
from adshield.models.comparison import (
    RESULTS_END,
    RESULTS_START,
    readme_results_section,
    render_report,
    update_readme,
    verdict,
)
from adshield.models.rule_engine import RuleEngine
from tests.synthetic import labelled_traffic


@pytest.fixture(scope="module")
def clicks() -> pd.DataFrame:
    return labelled_traffic()


@pytest.fixture(scope="module")
def small_config() -> ModelConfig:
    cfg = ModelConfig.from_yaml()
    raw = cfg.model_dump()
    raw["xgboost"]["grid"] = {"max_depth": [2, 4], "learning_rate": [0.2]}
    raw["xgboost"]["fixed"]["n_estimators"] = 60
    raw["xgboost"]["fixed"]["early_stopping_rounds"] = 10
    raw["shap"]["sample_size"] = 200
    return ModelConfig.model_validate(raw)


@pytest.fixture(scope="module")
def run(clicks, small_config, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    return train_and_evaluate(
        clicks, small_config, RuleEngine.from_yaml(), models_dir=out, version="vtest", data_source="synthetic"
    )


# ---- time split ----------------------------------------------------------------


def test_time_split_is_chronological_and_disjoint(clicks) -> None:
    sp = time_split(clicks, 0.6, 0.2)
    t = clicks["click_time"]
    assert t[sp.train].max() < sp.val_start <= t[sp.val].min()
    assert t[sp.val].max() < sp.test_start <= t[sp.test].min()
    all_idx = sp.train.append(sp.val).append(sp.test)
    assert len(all_idx) == len(clicks) and all_idx.is_unique
    assert abs(len(sp.train) / len(clicks) - 0.6) < 0.05


def test_time_split_ignores_row_order(clicks) -> None:
    shuffled = clicks.sample(frac=1, random_state=1)
    a, b = time_split(clicks, 0.6, 0.2), time_split(shuffled, 0.6, 0.2)
    assert set(a.test) == set(b.test) and a.val_start == b.val_start


def test_time_split_keeps_identical_timestamps_together() -> None:
    t0 = pd.Timestamp("2017-11-06")
    df = pd.DataFrame(
        {"click_time": [t0 + pd.Timedelta(seconds=s) for s in [0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8]]}
    )
    sp = time_split(df, 0.5, 0.25)
    times = df["click_time"]
    for a, b in ((sp.train, sp.val), (sp.val, sp.test)):
        assert set(times[a]).isdisjoint(set(times[b]))


def test_time_split_rejects_bad_input() -> None:
    df = pd.DataFrame({"click_time": pd.Timestamp("2017-11-06")}, index=[0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="unique"):
        time_split(df, 0.6, 0.2)
    same = pd.DataFrame({"click_time": [pd.Timestamp("2017-11-06")] * 20})
    with pytest.raises(ValueError, match="empty"):
        time_split(same, 0.6, 0.2)
    with pytest.raises(ValueError, match="at least"):
        time_split(same.iloc[:5], 0.6, 0.2)


# ---- imbalance, grid, metrics --------------------------------------------------


def test_scale_pos_weight() -> None:
    assert scale_pos_weight(np.array([1, 0, 0, 0, 0])) == pytest.approx(4.0)
    assert scale_pos_weight(np.array([1, 1, 1, 0])) == pytest.approx(1 / 3)  # majority positive
    with pytest.raises(ValueError):
        scale_pos_weight(np.array([1, 1]))


def test_grid_trials_is_full_deterministic_product() -> None:
    trials = grid_trials({"b": [1, 2], "a": [10, 20, 30]})
    assert len(trials) == 6 and trials[0] == {"a": 10, "b": 1} and trials[-1] == {"a": 30, "b": 2}


def test_evaluate_scores_perfect_and_random() -> None:
    y = np.array([0] * 90 + [1] * 10)
    perfect = evaluate_scores(y, y.astype(float), 0.5)
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert perfect["target_view"]["f1"] == 1.0 and perfect["pr_auc_baseline"] == pytest.approx(0.1)
    rng = np.random.default_rng(0)
    rand = evaluate_scores(y, rng.random(100), 0.5)
    assert rand["pr_auc"] < 0.4  # near the 0.10 floor, far below ROC-AUC-style optimism of a good model


def test_pr_auc_minority_flips_when_fraud_is_majority() -> None:
    y = np.array([1] * 90 + [0] * 10)  # fraud-proxy is the majority
    scores = np.where(y == 1, 0.9, 0.1)
    m = evaluate_scores(y, scores, 0.5)
    assert not m["minority_is_target"] and m["pr_auc_minority"] == 1.0
    assert m["minority_prevalence"] == pytest.approx(0.1)


def test_evaluate_scores_hand_computed_confusion() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    m = evaluate_scores(y, s, 0.5)
    assert m["confusion"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["target_view"]["precision"] == pytest.approx(2 / 3)


def test_evaluate_scores_single_class_raises() -> None:
    with pytest.raises(ValueError, match="both classes"):
        evaluate_scores(np.zeros(5), np.arange(5.0), 0.5)


def test_tune_threshold_separates_clean_data() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert 0.3 < tune_threshold(y, p) <= 0.7


def test_tune_threshold_avoids_flag_everything_when_fraud_is_majority() -> None:
    # 90% "fraud"; minority (10%) has low scores. Target-F1 would pick flag-all; we must not.
    y = np.array([1] * 90 + [0] * 10)
    p = np.concatenate([np.linspace(0.6, 0.95, 90), np.linspace(0.05, 0.4, 10)])
    th = tune_threshold(y, p)
    assert 0.4 < th <= 0.6
    assert evaluate_scores(y, p, th)["minority_view"]["f1"] == 1.0


def test_tune_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(5)
    y = rng.random(300) < 0.3
    s = np.round(rng.random(300) + y * 0.4, 2)
    th = tune_threshold(y, s)
    best = max(evaluate_scores(y, s, c)["minority_view"]["f1"] for c in np.unique(s))
    assert evaluate_scores(y, s, th)["minority_view"]["f1"] == pytest.approx(best)


# ---- end-to-end run -------------------------------------------------------------


def test_every_trial_is_logged_and_recorded(clicks, small_config, tmp_path, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="adshield.models.classifier"):
        r = train_and_evaluate(clicks, small_config, RuleEngine.from_yaml(), tmp_path, "vlog")
    trial_logs = [m for m in caplog.messages if m.startswith("trial ")]
    assert len(trial_logs) == len(r.trials) == 2
    assert all("params=" in m and "val_pr_auc_minority" in m for m in trial_logs)
    saved = json.loads((r.directory / "trials.json").read_text())
    assert saved == json.loads(json.dumps(r.trials))
    assert sum(t["selected"] for t in saved) == 1
    best = max(saved, key=lambda t: t["val_pr_auc_minority"])
    assert best["selected"]


def test_artifacts_and_metrics_json(run) -> None:
    d = run.directory
    for name in ("model.json", "pipeline.joblib", "metrics.json", "trials.json", "shap_importance.csv"):
        assert (d / name).exists()
    saved = json.loads((d / "metrics.json").read_text())
    assert saved["version"] == "vtest" and saved["data_source"] == "synthetic"
    assert saved["test"]["model"]["pr_auc"] == pytest.approx(run.metrics["test"]["model"]["pr_auc"])
    assert (
        saved["data"]["train_rows"] + saved["data"]["val_rows"] + saved["data"]["test_rows"]
        == saved["data"]["n_rows"]
    )
    assert set(saved["artifacts"]) == {"model.json", "pipeline.joblib", "trials.json", "shap_importance.csv"}
    assert pd.Timestamp(saved["data"]["val_start"]) < pd.Timestamp(saved["data"]["test_start"])


def test_existing_version_is_not_overwritten(clicks, small_config, run) -> None:
    with pytest.raises(FileExistsError):
        train_and_evaluate(clicks, small_config, RuleEngine.from_yaml(), run.directory.parent, "vtest")


def test_model_beats_chance_on_held_out_period(run) -> None:
    m = run.metrics["test"]["model"]
    assert m["roc_auc"] > 0.8
    assert m["pr_auc_minority"] > m["minority_prevalence"] + 0.1


def test_loaded_model_reproduces_predictions(run, clicks) -> None:
    loaded = load_model(run.directory)
    feats = run.model.pipeline.transform(clicks)
    np.testing.assert_allclose(loaded.predict_proba(feats), run.model.predict_proba(feats), rtol=1e-6)
    assert loaded.threshold == run.model.threshold


def test_load_detects_tampered_artifact(run, tmp_path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(run.directory, copy)
    (copy / "trials.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        load_model(copy)


def test_train_features_use_out_of_fold_risk(run, clicks, small_config) -> None:
    pipe = run.model.pipeline
    sp = time_split(clicks, 0.6, 0.2)
    y = clicks["is_attributed"] == 0
    naive = pipe.transform(clicks).loc[sp.train, pipe.encoder.feature_name]
    oof = pipe.transform(clicks, train_labels=y[sp.train])[pipe.encoder.feature_name].loc[sp.train]
    assert not np.allclose(naive, oof)  # in-sample encoding differs, i.e. would have leaked


# ---- SHAP / explanations ---------------------------------------------------------


def test_shap_values_sum_to_model_margin(run, clicks) -> None:
    feats = run.model.pipeline.transform(clicks).iloc[:50]
    shap_vals = run.model.shap_values(feats)
    base = float(np.ravel(run.model.explainer().expected_value)[0])
    margin = run.model.booster.predict(feats[run.model.feature_names], output_margin=True)
    np.testing.assert_allclose(shap_vals.sum(axis=1) + base, margin, atol=1e-4)


def test_explain_returns_ranked_top_contributors(run, clicks) -> None:
    feats = run.model.pipeline.transform(clicks).iloc[:20]
    ex = run.model.explain(feats, top_k=3)
    assert len(ex) == 20 * 3 and set(ex["rank"]) == {1, 2, 3}
    first = ex[ex["row"] == feats.index[0]]
    assert (first["shap"].abs().diff().dropna() <= 1e-12).all()  # descending |shap|
    assert set(first["feature"]) <= set(run.model.feature_names)
    assert ((first["shap"] > 0) == (first["direction"] == "raises_risk")).all()


def test_global_importance_is_ranked_and_saved(run) -> None:
    imp = run.shap_importance
    assert set(imp["feature"]) == set(run.model.feature_names)
    assert imp["mean_abs_shap"].is_monotonic_decreasing and imp["mean_abs_shap"].iloc[0] > 0
    assert (run.directory / "shap_importance.csv").exists()


# ---- fallback layer --------------------------------------------------------------


def test_scorer_uses_model_when_healthy(run, clicks) -> None:
    feats = run.model.pipeline.transform(clicks).iloc[:30]
    out = FraudScorer(run.model, RuleEngine.from_yaml()).score(feats)
    assert (out["source"] == "model").all() and out["model_probability"].notna().all()
    assert "fired_rules" in out and "rule_score" in out


def test_scorer_falls_back_without_model(run, clicks) -> None:
    feats = run.model.pipeline.transform(clicks).iloc[:30]
    out = FraudScorer(None, RuleEngine.from_yaml()).score(feats)
    assert (out["source"] == "rules_fallback").all() and (out["fallback_reason"] == "no_model").all()
    assert out["model_probability"].isna().all()


def test_scorer_falls_back_per_row_on_invalid_probability(run, clicks) -> None:
    feats = run.model.pipeline.transform(clicks).iloc[:10]

    class Broken:
        threshold = 0.5

        def predict_proba(self, f):
            p = np.full(len(f), 0.9)
            p[3] = np.nan
            p[4] = 1.5
            return p

    out = FraudScorer(Broken(), RuleEngine.from_yaml()).score(feats)
    assert out["source"].tolist().count("rules_fallback") == 2
    assert out["fallback_reason"].iloc[3] == "invalid_model_output"
    assert out["source"].iloc[0] == "model"


def test_scorer_falls_back_on_model_exception(run, clicks) -> None:
    feats = run.model.pipeline.transform(clicks).iloc[:5]

    class Exploding:
        threshold = 0.5

        def predict_proba(self, f):
            raise RuntimeError("boom")

    out = FraudScorer(Exploding(), RuleEngine.from_yaml()).score(feats)
    assert (out["fallback_reason"] == "model_error").all()


# ---- comparison report -----------------------------------------------------------


def _m(ap: float, f1: float, n_rows: int = 10_000) -> dict:
    return {"pr_auc_minority": ap, "minority_view": {"f1": f1}, "minority_prevalence": 0.1, "n_rows": n_rows}


@pytest.mark.parametrize(
    "model, rules, expected",
    [
        (_m(0.9, 0.9), _m(0.5, 0.5), "clearly_better"),
        (_m(0.9, 0.51), _m(0.5, 0.5), "not_clearly_better"),  # AP up, F1 flat
        (_m(0.51, 0.52), _m(0.5, 0.5), "not_clearly_better"),  # within margin
        (_m(0.4, 0.9), _m(0.5, 0.5), "worse"),
    ],
)
def test_verdict_does_not_overstate(model, rules, expected) -> None:
    assert verdict(model, rules, 0.02) == expected


def test_verdict_is_inconclusive_on_small_test_sets() -> None:
    small = _m(0.9, 0.9, n_rows=100)  # 10 minority rows
    assert verdict(small, _m(0.5, 0.5, n_rows=100), 0.02, min_minority_rows=500) == "inconclusive"
    assert verdict(_m(0.9, 0.9), _m(0.5, 0.5), 0.02, min_minority_rows=500) == "clearly_better"


def test_report_states_honest_verdict(run) -> None:
    text = render_report(run.metrics)
    assert "# Model vs rule-baseline comparison" in text
    assert "Flag every click" in text and "synthetic" in text
    assert "validation-tuned" in text and "configured threshold" in text
    assert "Inconclusive" in text  # synthetic test set is far below the minimum minority rows


def test_readme_results_use_the_runs_exact_numbers(run, tmp_path) -> None:
    section = readme_results_section(run.metrics)
    test = run.metrics["test"]
    for key in ("model", "rules_tuned", "rules", "flag_all"):
        t = test[key]["target_view"]
        assert (
            f"{t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} | {test[key]['pr_auc']:.4f}" in section
        )
    assert run.metrics["version"] in section and "Inconclusive" in section  # tiny synthetic test set


def test_update_readme_replaces_only_the_marked_block_and_is_idempotent(run, tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\nbefore\n{RESULTS_START}\nold numbers\n{RESULTS_END}\nafter\n")
    update_readme(readme, run.metrics)
    first = readme.read_text()
    assert "old numbers" not in first and first.startswith("# T\nbefore\n") and first.endswith("after\n")
    update_readme(readme, run.metrics)
    assert readme.read_text() == first


def test_update_readme_without_markers_raises(run, tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# no markers\n")
    with pytest.raises(ValueError, match="markers"):
        update_readme(readme, run.metrics)
