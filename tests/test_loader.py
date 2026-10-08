"""Tests for adshield.data.loader using small synthetic CSVs (never real data)."""

from pathlib import Path

import pandas as pd
import pytest

from adshield.data.loader import (
    COLUMNS,
    DatasetValidationError,
    dataset_health,
    load_clicks,
)

HEADER = ",".join(COLUMNS)
GOOD_ROWS = [
    "83230,3,1,13,379,2017-11-06 14:32:21,,0",
    "17357,3,1,19,379,2017-11-06 14:33:34,,0",
    "35810,3,1,13,379,2017-11-07 18:00:00,2017-11-07 19:10:00,1",
    "45745,14,1,13,478,2017-11-08 09:30:05,,0",
]


def write_csv(tmp_path: Path, rows: list[str], header: str = HEADER) -> Path:
    path = tmp_path / "clicks.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


@pytest.fixture
def good_csv(tmp_path: Path) -> Path:
    return write_csv(tmp_path, GOOD_ROWS)


def test_load_applies_explicit_dtypes(good_csv: Path) -> None:
    df = load_clicks(good_csv)
    assert list(df.columns) == list(COLUMNS)
    for col in ("ip", "app", "device", "os", "channel"):
        assert df[col].dtype == "int32"
    assert df["is_attributed"].dtype == "int8"
    assert pd.api.types.is_datetime64_any_dtype(df["click_time"])
    assert pd.api.types.is_datetime64_any_dtype(df["attributed_time"])
    assert df["attributed_time"].isna().sum() == 3


def test_nrows_limits_rows(good_csv: Path) -> None:
    assert len(load_clicks(good_csv, nrows=2)) == 2


def test_health_report(good_csv: Path) -> None:
    health = dataset_health(load_clicks(good_csv))
    assert health.n_rows == 4
    assert health.n_real == 1
    assert health.n_fraud == 3
    assert health.pct_real == pytest.approx(25.0)
    assert health.pct_fraud == pytest.approx(75.0)
    assert health.click_time_min == pd.Timestamp("2017-11-06 14:32:21")
    assert health.click_time_max == pd.Timestamp("2017-11-08 09:30:05")
    assert health.pct_missing["attributed_time"] == pytest.approx(75.0)
    assert health.pct_missing["ip"] == 0.0
    assert "fraud" in str(health)


@pytest.mark.parametrize(
    "bad_row, match",
    [
        ("83230,3,1,13,379,2017-11-06 14:32:21,,2", r"is_attributed"),  # label out of range
        ("-5,3,1,13,379,2017-11-06 14:32:21,,0", r"row 4 \[ip\]"),  # negative code
        ("83230,99999,1,13,379,2017-11-06 14:32:21,,0", r"\[app\]"),  # code above bound
        ("83230,3,1,13,99999,2017-11-06 14:32:21,,0", r"\[channel\]"),
        ("83230,3,1,13,379,1999-01-01 00:00:00,,0", r"click_time"),  # timestamp out of range
        ("83230,3,1,13,379,2017-11-06 14:32:21,2017-11-06 15:00:00,0", r"to be null"),
        ("83230,3,1,13,379,2017-11-06 14:32:21,,1", r"requires attributed_time"),
        (
            "83230,3,1,13,379,2017-11-06 14:32:21,2017-11-05 10:00:00,1",
            r"attributed_time",  # conversion before the click
        ),
    ],
)
def test_validation_rejects_bad_rows(tmp_path: Path, bad_row: str, match: str) -> None:
    path = write_csv(tmp_path, [*GOOD_ROWS, bad_row])
    with pytest.raises(DatasetValidationError, match=match):
        load_clicks(path)


def test_validation_reports_row_position(tmp_path: Path) -> None:
    bad = "83230,99999,1,13,379,2017-11-06 14:32:21,,0"
    path = write_csv(tmp_path, [*GOOD_ROWS, bad])
    with pytest.raises(DatasetValidationError, match=r"row 4 \[app\]"):
        load_clicks(path)


def test_validate_false_skips_row_checks(tmp_path: Path) -> None:
    path = write_csv(tmp_path, [*GOOD_ROWS, "83230,99999,1,13,379,2017-11-06 14:32:21,,0"])
    assert len(load_clicks(path, validate=False)) == 5


def test_null_in_required_field_fails(tmp_path: Path) -> None:
    path = write_csv(tmp_path, [*GOOD_ROWS, "83230,,1,13,379,2017-11-06 14:32:21,,0"])
    with pytest.raises(DatasetValidationError):
        load_clicks(path)


def test_null_click_time_fails(tmp_path: Path) -> None:
    path = write_csv(tmp_path, [*GOOD_ROWS, "83230,3,1,13,379,,,0"])
    with pytest.raises(DatasetValidationError, match="click_time"):
        load_clicks(path)


def test_non_numeric_code_fails(tmp_path: Path) -> None:
    path = write_csv(tmp_path, [*GOOD_ROWS, "abc,3,1,13,379,2017-11-06 14:32:21,,0"])
    with pytest.raises(DatasetValidationError, match="declared schema"):
        load_clicks(path)


def test_malformed_timestamp_fails(tmp_path: Path) -> None:
    path = write_csv(tmp_path, [*GOOD_ROWS, "83230,3,1,13,379,not-a-date,,0"])
    with pytest.raises(DatasetValidationError):
        load_clicks(path)


def test_missing_column_fails(tmp_path: Path) -> None:
    header = ",".join(c for c in COLUMNS if c != "channel")
    path = write_csv(tmp_path, ["1,2,3,4,2017-11-06 14:32:21,,0"], header=header)
    with pytest.raises(DatasetValidationError, match="columns"):
        load_clicks(path)


def test_empty_dataset_health_fails(tmp_path: Path) -> None:
    df = load_clicks(write_csv(tmp_path, GOOD_ROWS)).iloc[0:0]
    with pytest.raises(DatasetValidationError, match="empty"):
        dataset_health(df)
