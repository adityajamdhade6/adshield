"""Load the TalkingData AdTracking Fraud Detection click dataset.

Provides an explicitly typed CSV loader, strict per-row Pydantic validation
that raises on any violation (no rows are dropped or coerced), and a dataset
health report.

Label semantics: ``is_attributed`` == 1 is a real conversion and 0 is a click
that never converted. The latter is used as a practical fraud-risk proxy, not
a ground-truth fraud label.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

CSV_DTYPES: Final[dict[str, str]] = {
    "ip": "int32",
    "app": "int32",
    "device": "int32",
    "os": "int32",
    "channel": "int32",
    "is_attributed": "int8",
}
DATETIME_COLUMNS: Final[tuple[str, ...]] = ("click_time", "attributed_time")
DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
COLUMNS: Final[tuple[str, ...]] = (
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "click_time",
    "attributed_time",
    "is_attributed",
)

# Upper bounds on the categorical codes (inclusive), and the click window.
# Taken from the published train.csv; confirm against the real file on first
# load, where a mismatch fails loudly.
IP_MAX: Final[int] = 364_778
APP_MAX: Final[int] = 768
DEVICE_MAX: Final[int] = 4_227
OS_MAX: Final[int] = 956
CHANNEL_MAX: Final[int] = 498
TIME_MIN: Final[datetime] = datetime(2017, 11, 1)
TIME_MAX: Final[datetime] = datetime(2017, 12, 1)

VALIDATION_CHUNK_ROWS: Final[int] = 100_000
MAX_REPORTED_ERRORS: Final[int] = 10


class DatasetValidationError(ValueError):
    """Raised when the input file or any of its rows violates the schema."""


class ClickRecord(BaseModel):
    """One click row. Strict mode: values are never coerced to fit."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    ip: int = Field(ge=0, le=IP_MAX)
    app: int = Field(ge=0, le=APP_MAX)
    device: int = Field(ge=0, le=DEVICE_MAX)
    os: int = Field(ge=0, le=OS_MAX)
    channel: int = Field(ge=0, le=CHANNEL_MAX)
    click_time: datetime
    attributed_time: datetime | None
    is_attributed: int = Field(ge=0, le=1)

    def model_post_init(self, __context: object) -> None:
        if not TIME_MIN <= self.click_time < TIME_MAX:
            raise ValueError(f"click_time {self.click_time} outside [{TIME_MIN}, {TIME_MAX})")
        if self.is_attributed == 1:
            if self.attributed_time is None:
                raise ValueError("is_attributed=1 requires attributed_time")
            if not self.click_time <= self.attributed_time < TIME_MAX:
                raise ValueError(
                    f"attributed_time {self.attributed_time} must be within [click_time, {TIME_MAX})"
                )
        elif self.attributed_time is not None:
            raise ValueError("is_attributed=0 requires attributed_time to be null")


_RECORDS_ADAPTER: Final = TypeAdapter(list[ClickRecord])


def load_clicks(
    path: str | Path,
    *,
    nrows: int | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Read the raw click CSV with an explicit dtype schema.

    Raises ``DatasetValidationError`` if columns are missing or unexpected,
    values cannot be parsed to the declared types, or (when ``validate`` is
    true) any row fails ``ClickRecord`` validation.
    """
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if sorted(header) != sorted(COLUMNS):
        raise DatasetValidationError(f"{path}: columns {header} do not match expected {list(COLUMNS)}")

    try:
        df = pd.read_csv(
            path,
            usecols=list(COLUMNS),
            dtype=CSV_DTYPES,
            parse_dates=list(DATETIME_COLUMNS),
            date_format=DATETIME_FORMAT,
            nrows=nrows,
        )
    except ValueError as exc:
        raise DatasetValidationError(f"{path}: could not parse with declared schema: {exc}") from exc

    for col in DATETIME_COLUMNS:
        if df[col].dtype.kind != "M":
            raise DatasetValidationError(
                f"{path}: column {col!r} has values not matching format {DATETIME_FORMAT!r}"
            )

    df = df[list(COLUMNS)]
    if validate:
        validate_clicks(df)
    return df


def validate_clicks(df: pd.DataFrame) -> None:
    """Validate every row against ``ClickRecord``; raise on the first failing chunk.

    Nothing is dropped or coerced. The error message lists up to
    ``MAX_REPORTED_ERRORS`` offending rows with their row positions.
    """
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise DatasetValidationError(f"missing columns: {sorted(missing)}")

    required = [c for c in COLUMNS if c != "attributed_time"]
    null_counts = df[required].isna().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        raise DatasetValidationError(f"nulls in required columns: {bad}")

    for start in range(0, len(df), VALIDATION_CHUNK_ROWS):
        chunk = df.iloc[start : start + VALIDATION_CHUNK_ROWS]
        records = chunk[list(COLUMNS)].to_dict("records")
        for rec in records:
            if rec["attributed_time"] is pd.NaT:
                rec["attributed_time"] = None
        try:
            _RECORDS_ADAPTER.validate_python(records)
        except ValidationError as exc:
            lines = []
            errors = exc.errors()
            for err in errors[:MAX_REPORTED_ERRORS]:
                row = start + int(err["loc"][0])
                field = ".".join(str(p) for p in err["loc"][1:]) or "<row>"
                lines.append(f"  row {row} [{field}]: {err['msg']} (input={err['input']!r})")
            more = len(errors) - MAX_REPORTED_ERRORS
            if more > 0:
                lines.append(f"  ... and {more} more in this chunk")
            raise DatasetValidationError(
                f"{len(errors)} schema violation(s) in rows {start}-{start + len(chunk) - 1}:\n"
                + "\n".join(lines)
            ) from exc


@dataclass(frozen=True)
class DatasetHealth:
    """Summary statistics for a loaded click dataset."""

    n_rows: int
    n_real: int
    n_fraud: int
    pct_real: float
    pct_fraud: float
    click_time_min: pd.Timestamp
    click_time_max: pd.Timestamp
    pct_missing: dict[str, float]

    def __str__(self) -> str:
        lines = [
            f"rows:           {self.n_rows:,}",
            f"real (label 1): {self.n_real:,} ({self.pct_real:.4f}%)",
            f"fraud (label 0): {self.n_fraud:,} ({self.pct_fraud:.4f}%)",
            f"click_time:     {self.click_time_min} -> {self.click_time_max}",
            "missing per column (%):",
        ]
        lines += [f"  {col:<16}{pct:>10.4f}" for col, pct in self.pct_missing.items()]
        return "\n".join(lines)


def dataset_health(df: pd.DataFrame) -> DatasetHealth:
    """Compute row count, class balance, date range and per-column missingness."""
    if df.empty:
        raise DatasetValidationError("cannot compute health of an empty dataset")
    n_rows = len(df)
    n_real = int((df["is_attributed"] == 1).sum())
    n_fraud = n_rows - n_real
    return DatasetHealth(
        n_rows=n_rows,
        n_real=n_real,
        n_fraud=n_fraud,
        pct_real=100.0 * n_real / n_rows,
        pct_fraud=100.0 * n_fraud / n_rows,
        click_time_min=df["click_time"].min(),
        click_time_max=df["click_time"].max(),
        pct_missing={col: float(100.0 * df[col].isna().mean()) for col in df.columns},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a click CSV and print its health report.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--nrows", type=int, default=None, help="only read the first N rows")
    args = parser.parse_args()
    print(dataset_health(load_clicks(args.path, nrows=args.nrows)))


if __name__ == "__main__":
    main()
