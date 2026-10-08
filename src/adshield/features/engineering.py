"""Production feature engineering for click-fraud detection.

All functions are pure: they take a click DataFrame (columns ``ip``, ``app``,
``device``, ``os``, ``channel``, ``click_time``), never mutate it, and return
a new DataFrame sharing the input's index and row order.

Every feature is point-in-time: the value for a click uses only that click
and earlier events (plus events with an identical timestamp for the windowed
counts). Nothing looks at later clicks, so features computed on a full
dataset match what is computable at scoring time and do not leak the future.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

ENTITY_COLUMNS: Final[tuple[str, ...]] = ("ip", "app", "device", "os", "channel")
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (*ENTITY_COLUMNS, "click_time")

# Feature name -> trailing window length in seconds.
DEFAULT_VELOCITY_WINDOWS: Final[Mapping[str, int]] = {
    "clicks_ip_10min": 10 * 60,
    "clicks_ip_1h": 60 * 60,
    "clicks_ip_24h": 24 * 60 * 60,
}

_EPOCH: Final = pd.Timestamp("1970-01-01")


def _check_input(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df[list(columns)].isna().any().any():
        raise ValueError("input contains nulls in required columns")


def _epoch_seconds(click_time: pd.Series) -> np.ndarray:
    """Click times as int64 seconds since the Unix epoch."""
    if not pd.api.types.is_datetime64_any_dtype(click_time):
        raise ValueError("click_time must be a datetime column")
    return ((click_time - _EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype="int64")


def _order_by_ip_then_time(ip_codes: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    """Stable row order grouping each IP's clicks in time order (ties keep input order)."""
    return np.lexsort((seconds, ip_codes))


def velocity_features(
    df: pd.DataFrame,
    windows: Mapping[str, int] = DEFAULT_VELOCITY_WINDOWS,
) -> pd.DataFrame:
    """Clicks from the same IP in trailing time windows.

    For each click at time ``t`` and window ``w`` seconds, the feature is the
    number of clicks from the same IP with time in the half-open interval
    ``(t - w, t]``. The click itself is counted, so the minimum value is 1.

    Why this is a fraud signal: a person generates a handful of ad clicks per
    hour at most, limited by reading, page loads and attention. Click farms,
    scripts and botnets are limited only by bandwidth, so they produce bursts
    that no human session can. Three windows separate a short burst (10 min)
    from sustained abuse (1 h, 24 h); a shared NAT/office IP raises the
    longer windows without raising the short one, which helps tell the two
    apart.

    Args:
        df: Click events.
        windows: Mapping of output column name to window length in seconds.
    """
    if not windows or any(w <= 0 for w in windows.values()):
        raise ValueError("windows must be a non-empty mapping of positive seconds")
    _check_input(df, ("ip", "click_time"))
    if df.empty:
        return pd.DataFrame({name: np.empty(0, "int32") for name in windows}, index=df.index)

    seconds = _epoch_seconds(df["click_time"])
    seconds = seconds - seconds.min()
    ip_codes = pd.factorize(df["ip"])[0].astype("int64")

    # Offset IP blocks far enough apart that a window never reaches a neighbouring IP.
    offset = int(seconds.max()) + max(windows.values()) + 1
    keys = ip_codes * offset + seconds
    order = _order_by_ip_then_time(ip_codes, seconds)
    sorted_keys = keys[order]
    upper = np.searchsorted(sorted_keys, sorted_keys, side="right")

    out: dict[str, np.ndarray] = {}
    for name, width in windows.items():
        lower = np.searchsorted(sorted_keys, sorted_keys - width, side="right")
        counts = np.empty(len(df), dtype="int32")
        counts[order] = upper - lower
        out[name] = counts
    return pd.DataFrame(out, index=df.index)


def uniqueness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Diversity of what each IP has clicked, as of each click.

    Columns, all counted over the IP's clicks up to and including this one
    (in time order, ties in input order):

    - ``ip_click_count``: clicks so far.
    - ``ip_distinct_apps`` / ``ip_distinct_channels`` / ``ip_distinct_devices``:
      distinct values seen so far.
    - ``ip_unique_ratio``: distinct (app, device, os, channel) combinations
      divided by clicks so far. Near 0 means the IP repeats the same click
      over and over, which is typical of scripted traffic; 1 means every click
      was different.

    Why this is a fraud signal: genuine users browse, so over time they hit
    several apps/channels/devices, or at least vary what they click. A script
    replays one request, so it racks up clicks with almost no new distinct
    values: a low unique ratio at high click count is the footprint of
    repetition. The opposite pattern (one IP touching very many distinct
    apps/channels) indicates proxy or farm exit nodes pooling many users'
    or operators' traffic.
    """
    _check_input(df, REQUIRED_COLUMNS)
    names = (
        "ip_click_count",
        "ip_distinct_apps",
        "ip_distinct_channels",
        "ip_distinct_devices",
        "ip_unique_ratio",
    )
    if df.empty:
        empty = {n: np.empty(0, "float64" if n == "ip_unique_ratio" else "int32") for n in names}
        return pd.DataFrame(empty, index=df.index)

    seconds = _epoch_seconds(df["click_time"])
    ip_codes = pd.factorize(df["ip"])[0].astype("int64")
    order = _order_by_ip_then_time(ip_codes, seconds)

    s = df[list(ENTITY_COLUMNS)].iloc[order].reset_index(drop=True)
    s["_ip"] = ip_codes[order]
    groups = s["_ip"]

    def distinct_so_far(columns: list[str]) -> np.ndarray:
        first_seen = ~s.duplicated(["_ip", *columns])
        return first_seen.groupby(groups).cumsum().to_numpy()

    click_count = groups.groupby(groups).cumcount().to_numpy() + 1
    values = {
        "ip_click_count": click_count,
        "ip_distinct_apps": distinct_so_far(["app"]),
        "ip_distinct_channels": distinct_so_far(["channel"]),
        "ip_distinct_devices": distinct_so_far(["device"]),
    }
    combos = distinct_so_far(["app", "device", "os", "channel"])

    out: dict[str, np.ndarray] = {}
    for name, sorted_values in values.items():
        arr = np.empty(len(df), dtype="int32")
        arr[order] = sorted_values
        out[name] = arr
    ratio = np.empty(len(df), dtype="float64")
    ratio[order] = combos / click_count
    out["ip_unique_ratio"] = ratio
    return pd.DataFrame(out, index=df.index)[list(names)]


def regularity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Spread of the time gaps between an IP's consecutive clicks, as of each click.

    Columns (gaps are seconds between consecutive clicks from the same IP,
    in time order; ties keep input order):

    - ``ip_gap_count``: number of gaps observed so far (clicks so far - 1).
    - ``ip_gap_std_s``: sample standard deviation of those gaps, NaN until at
      least two gaps (three clicks) exist. NaN means "not enough evidence",
      deliberately not 0, so a new IP is never mistaken for a perfectly
      regular one.

    Why this is a fraud signal: human click timing is noisy; reading time,
    network latency and distraction give gaps with a wide spread. A cron job
    or a loop with a fixed ``sleep`` fires at near-constant intervals, so its
    gap standard deviation is close to zero. Evenly spaced traffic is hard
    for a bot author to avoid unless they deliberately add random jitter,
    which makes this a behavioural signature rather than a volume signature
    (it catches low-and-slow bots that velocity features miss). Always
    interpret it together with ``ip_gap_count``: a std of 0 over two gaps is
    weak evidence, over fifty gaps it is strong.
    """
    _check_input(df, ("ip", "click_time"))
    if df.empty:
        return pd.DataFrame(
            {"ip_gap_count": np.empty(0, "int32"), "ip_gap_std_s": np.empty(0, "float64")},
            index=df.index,
        )

    seconds = _epoch_seconds(df["click_time"])
    ip_codes = pd.factorize(df["ip"])[0].astype("int64")
    order = _order_by_ip_then_time(ip_codes, seconds)
    sorted_ip = ip_codes[order]
    sorted_t = seconds[order]

    gaps = np.diff(sorted_t, prepend=sorted_t[:1]).astype("float64")
    gaps[np.r_[True, sorted_ip[1:] != sorted_ip[:-1]]] = np.nan  # first click has no gap

    gap_series = pd.Series(gaps)
    # pandas' expanding std uses a numerically stable online update, so
    # constant gaps give ~0 instead of the noise a sum-of-squares formula adds.
    std_sorted = (
        gap_series.groupby(sorted_ip).expanding(min_periods=2).std().droplevel(0).sort_index().to_numpy()
    )
    count_sorted = gap_series.notna().groupby(sorted_ip).cumsum().to_numpy()

    count = np.empty(len(df), dtype="int32")
    std = np.empty(len(df), dtype="float64")
    count[order] = count_sorted
    std[order] = std_sorted
    return pd.DataFrame({"ip_gap_count": count, "ip_gap_std_s": std}, index=df.index)


@dataclass(frozen=True)
class RiskEncoder:
    """Smoothed historical fraud rate per key combination, learned from training data only."""

    keys: tuple[str, ...]
    smoothing: float
    prior: float
    table: pd.DataFrame  # columns: *keys, "risk"

    @property
    def feature_name(self) -> str:
        return "cat_risk_" + "_".join(self.keys)


def _check_fraud_labels(df: pd.DataFrame, is_fraud: pd.Series) -> np.ndarray:
    if len(is_fraud) != len(df) or not is_fraud.index.equals(df.index):
        raise ValueError("is_fraud must have the same index as the training frame")
    if is_fraud.isna().any():
        raise ValueError("is_fraud contains nulls")
    labels = is_fraud.to_numpy()
    if not np.isin(labels, (0, 1, False, True)).all():
        raise ValueError("is_fraud must contain only 0/1 or booleans")
    return labels.astype("float64")


def fit_risk_encoder(
    train: pd.DataFrame,
    is_fraud: pd.Series,
    keys: tuple[str, ...] = ("app", "channel", "os"),
    smoothing: float = 20.0,
) -> RiskEncoder:
    """Learn the fraud rate of each ``keys`` combination from labelled training clicks.

    The rate is smoothed toward the overall training fraud rate:
    ``(fraud + smoothing * prior) / (clicks + smoothing)``. A combination
    seen 3 times with 3 frauds is not trusted as 100% risky; one seen 50,000
    times is barely moved.

    Why this is a fraud signal: fraud is concentrated in particular traffic
    sources. A given publisher/channel/OS combination is often where click
    farms and bad ad networks route their traffic, so its historical outcome
    rate predicts the next click's outcome better than any single field.
    Combinations that rarely convert across thousands of clicks are
    low-quality inventory, whoever the individual IP is.

    Leakage: call this on the training split only. Do not use the encoding of
    the same rows it was fitted on as a training feature (each row's own
    label would feed its feature); use ``out_of_fold_risk`` for that.
    """
    if not keys:
        raise ValueError("keys must be non-empty")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    _check_input(train, keys)
    labels = _check_fraud_labels(train, is_fraud)
    if len(train) == 0:
        raise ValueError("cannot fit a risk encoder on an empty training set")

    prior = float(labels.mean())
    grouped = (
        train[list(keys)]
        .assign(_fraud=labels)
        .groupby(list(keys), observed=True)["_fraud"]
        .agg(fraud="sum", clicks="size")
    )
    risk = (grouped["fraud"] + smoothing * prior) / (grouped["clicks"] + smoothing)
    table = risk.rename("risk").reset_index()
    return RiskEncoder(keys=tuple(keys), smoothing=float(smoothing), prior=prior, table=table)


def apply_risk_encoder(df: pd.DataFrame, encoder: RiskEncoder) -> pd.Series:
    """Look up each row's smoothed fraud rate; unseen combinations get the training prior."""
    _check_input(df, encoder.keys)
    merged = df[list(encoder.keys)].merge(encoder.table, how="left", on=list(encoder.keys))
    risk = merged["risk"].fillna(encoder.prior).to_numpy()
    return pd.Series(risk, index=df.index, name=encoder.feature_name)


def out_of_fold_risk(
    train: pd.DataFrame,
    is_fraud: pd.Series,
    keys: tuple[str, ...] = ("app", "channel", "os"),
    smoothing: float = 20.0,
    n_folds: int = 5,
    seed: int = 0,
) -> pd.Series:
    """Risk feature for the training rows themselves, without target leakage.

    Rows are split into ``n_folds`` random folds; each fold is encoded with an
    encoder fitted on the other folds, so no row's label influences its own
    feature. Use this to train a model, and ``apply_risk_encoder`` with an
    encoder fitted on all training data for validation/test/production rows.

    Random folds ignore time order; for strictly time-forward validation fit
    on an earlier period and apply to a later one.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(train) < n_folds:
        raise ValueError("fewer rows than folds")
    _check_fraud_labels(train, is_fraud)  # validates alignment and values up front
    fold = np.random.default_rng(seed).permutation(len(train)) % n_folds
    out = np.empty(len(train), dtype="float64")
    name = "cat_risk_" + "_".join(keys)
    for k in range(n_folds):
        held = fold == k
        encoder = fit_risk_encoder(train[~held], is_fraud[~held], keys, smoothing)
        out[held] = apply_risk_encoder(train[held], encoder).to_numpy()
    return pd.Series(out, index=train.index, name=name)


def build_features(df: pd.DataFrame, encoder: RiskEncoder | None = None) -> pd.DataFrame:
    """All engineered features for ``df``, aligned to its index.

    The categorical risk column is included only when a fitted ``encoder`` is
    given, since it depends on training labels.
    """
    parts = [velocity_features(df), uniqueness_features(df), regularity_features(df)]
    if encoder is not None:
        parts.append(apply_risk_encoder(df, encoder).to_frame())
    return pd.concat(parts, axis=1)
