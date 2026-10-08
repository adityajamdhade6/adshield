"""Tests for the rule engine and baseline evaluation, using synthetic click sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from adshield.features.engineering import build_features
from adshield.models.evaluate import binary_metrics, evaluate_rule_baseline, threshold_sweep
from adshield.models.rule_engine import RuleEngine, RulesConfig

T0 = pd.Timestamp("2017-11-06 12:00:00")


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine.from_yaml()  # the real configs/rules.yaml


def click_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (ip, app, device, os, channel, seconds_after_T0)"""
    return pd.DataFrame(
        {
            "ip": [r[0] for r in rows],
            "app": [r[1] for r in rows],
            "device": [r[2] for r in rows],
            "os": [r[3] for r in rows],
            "channel": [r[4] for r in rows],
            "click_time": [T0 + pd.Timedelta(seconds=r[5]) for r in rows],
        }
    )


def synthetic_traffic() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns clicks, boolean mask of bot rows, and boolean mask of the final bot click.

    - bot IP 1: 60 identical clicks, exactly 5 s apart.
    - human IPs 10..29: 4 clicks each with irregular hour-scale gaps and varied apps/channels.
    """
    rows = [(1, 3, 1, 13, 379, 5 * i) for i in range(60)]
    rng = np.random.default_rng(7)
    for ip in range(10, 30):
        t = 0
        for _ in range(4):
            t += int(rng.integers(900, 7200))
            rows.append(
                (
                    ip,
                    int(rng.integers(1, 20)),
                    int(rng.integers(1, 4)),
                    int(rng.integers(1, 9)),
                    int(rng.integers(100, 300)),
                    t,
                )
            )
    df = click_frame(rows)
    is_bot = df["ip"] == 1
    last_bot = is_bot & (df.index == df.index[is_bot][-1])
    return df, is_bot, last_bot


def test_known_fraud_scores_high_and_legit_scores_low(engine: RuleEngine) -> None:
    df, is_bot, last_bot = synthetic_traffic()
    feats = build_features(df)
    feats["cat_risk_app_channel_os"] = 0.5  # neutral traffic source
    scores = engine.score(feats)

    # After warm-up the bot trips velocity + metronome + repetition rules.
    assert scores[last_bot].iloc[0] >= 80
    assert (scores[is_bot].iloc[15:] >= engine.config.decision_threshold).all()
    # Humans never reach the threshold; here they fire nothing at all.
    assert (scores[~is_bot] == 0).all()


def test_first_clicks_of_a_bot_are_not_flagged_without_evidence(engine: RuleEngine) -> None:
    df, is_bot, _ = synthetic_traffic()
    feats = build_features(df)
    feats["cat_risk_app_channel_os"] = 0.5
    # NaN / low counts mean no evidence yet, so the very first click scores 0.
    assert engine.score(feats)[is_bot].iloc[0] == 0


def test_ten_clicks_in_one_minute_fires_velocity_rule(engine: RuleEngine) -> None:
    df = click_frame([(7, i, 1, 1, i, 6 * i) for i in range(10)])  # varied, so only velocity can fire
    feats = build_features(df)
    feats["cat_risk_app_channel_os"] = 0.5
    result = engine.score_detailed(feats)
    assert result.contributions["velocity_10min_burst"].iloc[-1] == 35
    assert result.contributions["velocity_10min_burst"].iloc[:9].sum() == 0
    assert result.fired_rules().iloc[-1] == ("velocity_10min_burst", "metronomic_clicks")


def test_contributions_sum_to_raw_and_score_is_capped(engine: RuleEngine) -> None:
    df, _, last_bot = synthetic_traffic()
    feats = build_features(df)
    feats["cat_risk_app_channel_os"] = 0.9999  # also fires the traffic-source rule
    result = engine.score_detailed(feats)
    assert np.allclose(result.contributions.sum(axis=1), result.raw_points)
    assert (result.score <= engine.config.max_score).all()
    assert (result.score == result.raw_points.clip(upper=100)).all()
    row = result.contributions[last_bot].iloc[0]
    assert row["risky_traffic_source"] == 15


def test_score_cap_applies_when_points_exceed_max() -> None:
    cfg = RulesConfig.model_validate(
        {
            "version": 1,
            "max_score": 100,
            "decision_threshold": 50,
            "rules": [
                {
                    "name": "a",
                    "description": "a",
                    "points": 80,
                    "conditions": [{"feature": "x", "op": ">", "threshold": 0}],
                },
                {
                    "name": "b",
                    "description": "b",
                    "points": 70,
                    "conditions": [{"feature": "x", "op": ">=", "threshold": 1}],
                },
            ],
        }
    )
    result = RuleEngine(cfg).score_detailed(pd.DataFrame({"x": [0.0, 1.0]}))
    assert result.raw_points.tolist() == [0.0, 150.0]
    assert result.score.tolist() == [0.0, 100.0]


def test_nan_feature_never_fires_a_rule(engine: RuleEngine) -> None:
    feats = pd.DataFrame({f: [np.nan] for f in engine.required_features})
    assert engine.score(feats).iloc[0] == 0


def test_all_conditions_must_hold() -> None:
    cfg = RulesConfig.model_validate(
        {
            "version": 1,
            "max_score": 100,
            "decision_threshold": 10,
            "rules": [
                {
                    "name": "both",
                    "description": "d",
                    "points": 40,
                    "conditions": [
                        {"feature": "a", "op": ">=", "threshold": 5},
                        {"feature": "b", "op": "<", "threshold": 1},
                    ],
                }
            ],
        }
    )
    feats = pd.DataFrame({"a": [5, 5, 4], "b": [0.5, 1.0, 0.0]})
    assert RuleEngine(cfg).score(feats).tolist() == [40, 0, 0]


def test_audit_trail_records_which_rules_fired_and_why(engine: RuleEngine) -> None:
    df = click_frame([(7, i, 1, 1, i, 6 * i) for i in range(10)])
    feats = build_features(df)
    feats["cat_risk_app_channel_os"] = 0.5
    trail = engine.audit_trail(feats)
    last = trail[trail["row"] == df.index[-1]]
    assert set(last["rule"]) == {"velocity_10min_burst", "metronomic_clicks"}
    burst = last[last["rule"] == "velocity_10min_burst"].iloc[0]
    assert burst["points"] == 35
    assert "clicks_ip_10min=10 (>= 10)" in burst["evidence"]
    assert burst["risk_score"] == 60
    assert (trail["config_hash"] == engine.config_hash).all()
    # Rows with nothing fired have no audit entries.
    assert df.index[0] not in set(trail["row"])


def test_scoring_preserves_index_and_does_not_mutate_input(engine: RuleEngine) -> None:
    df, _, _ = synthetic_traffic()
    feats = build_features(df)
    feats["cat_risk_app_channel_os"] = 0.5
    feats.index = feats.index + 1000
    before = feats.copy(deep=True)
    scores = engine.score(feats)
    assert scores.index.equals(feats.index)
    pd.testing.assert_frame_equal(feats, before)


def test_scoring_is_deterministic(engine: RuleEngine) -> None:
    df, _, _ = synthetic_traffic()
    feats = build_features(df)
    feats["cat_risk_app_channel_os"] = 0.5
    pd.testing.assert_series_equal(engine.score(feats), engine.score(feats.copy()))


def test_missing_feature_column_raises(engine: RuleEngine) -> None:
    with pytest.raises(ValueError, match="missing"):
        engine.score(pd.DataFrame({"clicks_ip_10min": [1]}))


def test_config_hash_changes_with_weights(tmp_path: Path) -> None:
    base = yaml.safe_load(Path("configs/rules.yaml").read_text())
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(base))
    base["rules"][0]["points"] += 1
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(base))
    assert (
        RuleEngine.from_yaml("a.yaml", tmp_path).config_hash
        != RuleEngine.from_yaml("b.yaml", tmp_path).config_hash
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["rules"].append(dict(c["rules"][0])),  # duplicate name
        lambda c: c["rules"][0].update(points=0),
        lambda c: c["rules"][0].update(points=-5),
        lambda c: c["rules"][0]["conditions"][0].update(op="=="),
        lambda c: c["rules"][0].update(conditions=[]),
        lambda c: c["rules"][0].update(unexpected="x"),
        lambda c: c.update(decision_threshold=101),
        lambda c: c.update(rules=[]),
    ],
)
def test_invalid_config_is_rejected(mutate, tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path("configs/rules.yaml").read_text())
    mutate(cfg)
    with pytest.raises(ValidationError):
        RulesConfig.model_validate(cfg)


def test_binary_metrics_hand_computed() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = binary_metrics(y, p)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 4)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.confusion_matrix.to_numpy().tolist() == [[4, 1], [1, 2]]


def test_binary_metrics_zero_division_is_zero() -> None:
    m = binary_metrics(np.array([0, 0]), np.array([0, 0]))
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_threshold_sweep_is_monotone_in_flag_rate() -> None:
    scores = pd.Series([0, 20, 40, 60, 80, 100.0])
    y = pd.Series([0, 0, 0, 1, 1, 1])
    sweep = threshold_sweep(scores, y, [10, 50, 90])
    assert sweep["flag_rate"].tolist() == pytest.approx([5 / 6, 3 / 6, 1 / 6])
    assert sweep["precision"].tolist() == pytest.approx([3 / 5, 1.0, 1.0])
    assert sweep["recall"].tolist() == pytest.approx([1.0, 1.0, 1 / 3])


def test_end_to_end_baseline_evaluation_on_labelled_synthetic_traffic(engine: RuleEngine) -> None:
    # Bot clicks never convert (fraud proxy = 1); human clicks convert (0). Interleave in time
    # so both the train and test periods contain both kinds.
    rng = np.random.default_rng(11)
    rows, labels = [], []
    for bot_ip in range(1, 4):
        for i in range(80):
            rows.append((bot_ip, 3, 1, 13, 379, 5 * i + bot_ip * 100_000))
            labels.append(0)  # is_attributed=0 -> fraud proxy
    for ip in range(100, 160):
        t = 0
        for _ in range(6):
            t += int(rng.integers(900, 7200))
            rows.append(
                (
                    ip,
                    int(rng.integers(1, 20)),
                    int(rng.integers(1, 4)),
                    int(rng.integers(1, 9)),
                    int(rng.integers(100, 300)),
                    t,
                )
            )
            labels.append(1)
    df = click_frame(rows)
    df["is_attributed"] = labels
    report = evaluate_rule_baseline(df, engine, train_frac=0.5)
    assert report.rule_metrics.tp + report.rule_metrics.fn > 0
    assert report.rule_metrics.fp == 0  # humans never reach the threshold
    assert report.rule_metrics.precision == 1.0
    assert report.rule_metrics.recall > 0.5
    assert "velocity_10min_burst" in report.rule_fire_rate.index
    assert "Threshold sweep" in str(report)
