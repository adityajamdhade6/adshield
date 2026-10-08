"""Tests for adshield.features.engineering on hand-built and random synthetic data."""

import numpy as np
import pandas as pd
import pytest

from adshield.features.engineering import (
    apply_risk_encoder,
    build_features,
    fit_risk_encoder,
    out_of_fold_risk,
    regularity_features,
    uniqueness_features,
    velocity_features,
)

T0 = pd.Timestamp("2017-11-06 12:00:00")


def clicks(rows: list[tuple]) -> pd.DataFrame:
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


def test_velocity_window_boundaries() -> None:
    # Clicks at 0, 300, 600, 601 s. Window is (t-w, t], so a click exactly w old is excluded.
    df = clicks([(1, 1, 1, 1, 1, s) for s in (0, 300, 600, 601)])
    out = velocity_features(df)
    assert out["clicks_ip_10min"].tolist() == [1, 2, 2, 3]
    assert out["clicks_ip_1h"].tolist() == [1, 2, 3, 4]
    assert out["clicks_ip_24h"].tolist() == [1, 2, 3, 4]


def test_velocity_is_per_ip_and_keeps_input_order_and_index() -> None:
    df = clicks(
        [
            (2, 1, 1, 1, 1, 100),
            (1, 1, 1, 1, 1, 50),
            (1, 1, 1, 1, 1, 0),
            (2, 1, 1, 1, 1, 0),
            (1, 1, 1, 1, 1, 10),
        ]
    )
    df.index = [10, 11, 12, 13, 14]
    out = velocity_features(df, {"w": 60})
    # ip1 times 50,0,10 -> 3, 1, 2 clicks in window. ip2: t=100 -> 1 (t=0 is 100s old), t=0 -> 1.
    assert out["w"].tolist() == [1, 3, 1, 1, 2]
    assert out.index.tolist() == [10, 11, 12, 13, 14]


def test_velocity_counts_simultaneous_clicks() -> None:
    df = clicks([(1, 1, 1, 1, 1, 5)] * 3)
    assert velocity_features(df, {"w": 60})["w"].tolist() == [3, 3, 3]


def test_velocity_matches_brute_force_on_random_data() -> None:
    rng = np.random.default_rng(0)
    n = 400
    df = clicks(
        [
            (int(ip), 1, 1, 1, 1, int(s))
            for ip, s in zip(rng.integers(0, 8, n), rng.integers(0, 200_000, n), strict=True)
        ]
    )
    windows = {"a": 600, "b": 3600, "c": 86400}
    out = velocity_features(df, windows)
    t = (df["click_time"] - T0).dt.total_seconds().to_numpy()
    ip = df["ip"].to_numpy()
    for name, w in windows.items():
        expected = [int(((ip == ip[i]) & (t > t[i] - w) & (t <= t[i])).sum()) for i in range(n)]
        assert out[name].tolist() == expected


def test_uniqueness_hand_computed() -> None:
    df = clicks(
        [
            (1, 1, 1, 1, 1, 0),  # new everything
            (1, 1, 1, 1, 1, 10),  # exact repeat
            (1, 2, 1, 1, 1, 20),  # new app
            (1, 2, 1, 1, 7, 30),  # new channel
            (2, 5, 5, 5, 5, 0),
        ]
    )
    out = uniqueness_features(df)
    assert out["ip_click_count"].tolist() == [1, 2, 3, 4, 1]
    assert out["ip_distinct_apps"].tolist() == [1, 1, 2, 2, 1]
    assert out["ip_distinct_channels"].tolist() == [1, 1, 1, 2, 1]
    assert out["ip_distinct_devices"].tolist() == [1, 1, 1, 1, 1]
    assert out["ip_unique_ratio"].tolist() == pytest.approx([1.0, 0.5, 2 / 3, 3 / 4, 1.0])


def test_uniqueness_is_point_in_time_regardless_of_row_order() -> None:
    ordered = clicks([(1, 1, 1, 1, 1, 0), (1, 2, 1, 1, 1, 10), (1, 3, 1, 1, 1, 20)])
    shuffled = ordered.iloc[[2, 0, 1]]
    out = uniqueness_features(shuffled)
    # Row order is [t=20, t=0, t=10]; each row sees only clicks up to its own time.
    assert out["ip_distinct_apps"].tolist() == [3, 1, 2]
    assert out.index.tolist() == [2, 0, 1]


def test_scripted_ip_has_low_ratio() -> None:
    df = clicks([(9, 1, 1, 1, 1, s) for s in range(0, 100, 10)])
    assert uniqueness_features(df)["ip_unique_ratio"].iloc[-1] == pytest.approx(0.1)


def test_functions_do_not_mutate_input() -> None:
    df = clicks([(1, 1, 1, 1, 1, 0), (1, 2, 1, 1, 1, 5)])
    before = df.copy(deep=True)
    build_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_build_features_columns_and_alignment() -> None:
    df = clicks([(1, 1, 1, 1, 1, 0), (2, 1, 1, 1, 1, 5)])
    out = build_features(df)
    assert list(out.columns) == [
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
    ]
    assert out.index.equals(df.index)


def test_build_features_adds_risk_column_only_with_encoder() -> None:
    df = clicks([(1, 1, 1, 1, 1, 0), (2, 2, 1, 1, 2, 5)])
    enc = fit_risk_encoder(df, pd.Series([1, 0], index=df.index))
    assert "cat_risk_app_channel_os" not in build_features(df).columns
    assert "cat_risk_app_channel_os" in build_features(df, enc).columns


def test_empty_input_returns_empty_features() -> None:
    df = clicks([(1, 1, 1, 1, 1, 0)]).iloc[0:0]
    assert len(build_features(df)) == 0


@pytest.mark.parametrize("func", [velocity_features, uniqueness_features, regularity_features])
def test_missing_column_raises(func) -> None:
    df = clicks([(1, 1, 1, 1, 1, 0)]).drop(columns=["click_time"])
    with pytest.raises(ValueError, match="missing"):
        func(df)


def test_null_input_raises() -> None:
    df = clicks([(1, 1, 1, 1, 1, 0), (1, 1, 1, 1, 1, 5)])
    df.loc[1, "click_time"] = pd.NaT
    with pytest.raises(ValueError, match="nulls"):
        velocity_features(df)


def test_invalid_windows_raise() -> None:
    df = clicks([(1, 1, 1, 1, 1, 0)])
    with pytest.raises(ValueError):
        velocity_features(df, {"w": 0})


def test_ten_clicks_in_one_minute_is_a_velocity_burst() -> None:
    # 10 clicks, 6 s apart (54 s total): the tenth click sees all ten in its 10-minute window.
    df = clicks([(7, 1, 1, 1, 1, 6 * i) for i in range(10)])
    out = velocity_features(df)
    assert out["clicks_ip_10min"].tolist() == list(range(1, 11))
    assert out["clicks_ip_10min"].iloc[-1] >= 10


def test_regularity_evenly_spaced_bot_has_zero_std() -> None:
    df = clicks([(5, 1, 1, 1, 1, 30 * i) for i in range(8)])
    out = regularity_features(df)
    assert out["ip_gap_count"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert out["ip_gap_std_s"].iloc[:2].isna().all()  # <2 gaps: no evidence yet, not "0"
    assert out["ip_gap_std_s"].iloc[2:].to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_regularity_irregular_gaps_hand_computed() -> None:
    # times 0, 10, 40, 100 -> gaps 10, 30, 60. Sample std of [10,30] = 14.142..., of [10,30,60] = 25.166...
    df = clicks([(5, 1, 1, 1, 1, s) for s in (0, 10, 40, 100)])
    std = regularity_features(df)["ip_gap_std_s"]
    assert np.isnan(std.iloc[:2]).all()
    assert std.iloc[2] == pytest.approx(np.std([10, 30], ddof=1))
    assert std.iloc[3] == pytest.approx(np.std([10, 30, 60], ddof=1))


def test_regularity_is_per_ip_and_order_independent() -> None:
    rows = [(1, 1, 1, 1, 1, s) for s in (0, 10, 20, 30)] + [(2, 1, 1, 1, 1, s) for s in (0, 1, 50, 51)]
    df = clicks(rows)
    shuffled = df.iloc[[7, 2, 5, 0, 3, 6, 1, 4]]
    a = regularity_features(df)
    b = regularity_features(shuffled).loc[df.index]
    pd.testing.assert_frame_equal(a, b)
    assert a["ip_gap_std_s"].iloc[3] == pytest.approx(0.0, abs=1e-9)  # ip 1: gaps 10,10,10
    assert a["ip_gap_std_s"].iloc[7] == pytest.approx(np.std([1, 49, 1], ddof=1))  # ip 2


def test_regularity_matches_brute_force_on_random_data() -> None:
    rng = np.random.default_rng(3)
    n = 200
    df = clicks(
        [
            (int(i), 1, 1, 1, 1, int(s))
            for i, s in zip(rng.integers(0, 5, n), rng.integers(0, 5000, n), strict=True)
        ]
    )
    out = regularity_features(df)
    for i in range(n):
        ip, t = df["ip"].iloc[i], df["click_time"].iloc[i]
        # events up to and including row i in (time, input-order) order
        same = df[df["ip"] == ip].reset_index()
        same = same.sort_values(["click_time", "index"], kind="stable")
        upto = same[(same["click_time"] < t) | ((same["click_time"] == t) & (same["index"] <= i))]
        gaps = upto["click_time"].diff().dt.total_seconds().dropna().to_numpy()
        assert out["ip_gap_count"].iloc[i] == len(gaps)
        if len(gaps) >= 2:
            assert out["ip_gap_std_s"].iloc[i] == pytest.approx(np.std(gaps, ddof=1), abs=1e-6)
        else:
            assert np.isnan(out["ip_gap_std_s"].iloc[i])


def _risk_frame() -> tuple[pd.DataFrame, pd.Series]:
    # combo A (app1,ch1,os1): 10 clicks, 9 fraud. combo B (app2,ch2,os2): 10 clicks, 1 fraud.
    rows = [(i, 1, 1, 1, 1, i) for i in range(10)] + [(100 + i, 2, 1, 2, 2, i) for i in range(10)]
    df = clicks(rows)
    fraud = pd.Series([1] * 9 + [0] + [1] + [0] * 9, index=df.index)
    return df, fraud


def test_risk_encoder_smoothed_rates_hand_computed() -> None:
    df, fraud = _risk_frame()
    enc = fit_risk_encoder(df, fraud, smoothing=10.0)
    assert enc.prior == pytest.approx(0.5)
    risk = apply_risk_encoder(df, enc)
    assert risk.name == "cat_risk_app_channel_os"
    assert risk.iloc[0] == pytest.approx((9 + 10 * 0.5) / (10 + 10))  # 0.7
    assert risk.iloc[10] == pytest.approx((1 + 10 * 0.5) / (10 + 10))  # 0.3
    assert risk.index.equals(df.index)


def test_risk_encoder_unseen_combo_gets_training_prior() -> None:
    df, fraud = _risk_frame()
    enc = fit_risk_encoder(df, fraud)
    unseen = clicks([(1, 99, 1, 99, 99, 0)])
    assert apply_risk_encoder(unseen, enc).iloc[0] == pytest.approx(enc.prior)


def test_risk_encoder_smoothing_limits_small_sample_overconfidence() -> None:
    df = clicks([(1, 1, 1, 1, 1, 0)] * 3 + [(2, 2, 1, 2, 2, 0)] * 97)
    fraud = pd.Series([1] * 3 + [0] * 97, index=df.index)
    enc = fit_risk_encoder(df, fraud, smoothing=20.0)
    risk_small = apply_risk_encoder(df.iloc[:1], enc).iloc[0]
    assert risk_small == pytest.approx((3 + 20 * 0.03) / (3 + 20))
    assert risk_small < 0.5  # 3-for-3 frauds is not trusted as ~100% risk


def test_out_of_fold_risk_does_not_see_own_label() -> None:
    # One row is the only member of its combo and is fraud. In-sample encoding would
    # move it toward 1; out-of-fold must give it the prior of the other folds instead.
    df = clicks([(i, 1, 1, 1, 1, i) for i in range(20)] + [(99, 7, 1, 7, 7, 0)])
    fraud = pd.Series([0] * 20 + [1], index=df.index)
    oof = out_of_fold_risk(df, fraud, smoothing=0.0, n_folds=3, seed=1)
    in_sample = apply_risk_encoder(df, fit_risk_encoder(df, fraud, smoothing=0.0))
    assert in_sample.iloc[-1] == pytest.approx(1.0)  # leaks its own label
    assert oof.iloc[-1] == pytest.approx(0.0)  # unseen in other folds -> prior of rest (all legit)
    assert oof.index.equals(df.index)


def test_risk_encoder_input_validation() -> None:
    df, fraud = _risk_frame()
    with pytest.raises(ValueError, match="index"):
        fit_risk_encoder(df, fraud.reset_index(drop=True).set_axis(range(100, 120)))
    with pytest.raises(ValueError, match="0/1"):
        fit_risk_encoder(df, fraud * 2)
    with pytest.raises(ValueError, match="empty"):
        fit_risk_encoder(df.iloc[0:0], fraud.iloc[0:0])
    with pytest.raises(ValueError, match="keys"):
        fit_risk_encoder(df, fraud, keys=())
