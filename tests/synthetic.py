"""Synthetic labelled click traffic shared by model-layer tests."""

import numpy as np
import pandas as pd

T0 = pd.Timestamp("2017-11-06 00:00:00")


def labelled_traffic(n_bots: int = 25, n_humans: int = 250, seed: int = 0) -> pd.DataFrame:
    """Bots: bursts of identical, evenly spaced clicks that never convert.
    Humans: sparse irregular clicks across varied apps/channels; ~30% convert.
    Everything is spread over three days so every chronological split sees both kinds.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bots):
        start = int(rng.integers(0, 3 * 86400 - 3600))
        app, ch = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        for i in range(40):
            rows.append((1000 + b, app, 1, 13, ch, start + 5 * i, 0))
    for h in range(n_humans):
        t = int(rng.integers(0, 2 * 86400))
        for _ in range(int(rng.integers(2, 6))):
            t += int(rng.integers(600, 20000))
            if t >= 3 * 86400:
                break
            rows.append(
                (
                    5000 + h,
                    int(rng.integers(5, 40)),
                    int(rng.integers(1, 4)),
                    int(rng.integers(1, 9)),
                    int(rng.integers(10, 60)),
                    t,
                    int(rng.random() < 0.3),
                )
            )
    df = pd.DataFrame(rows, columns=["ip", "app", "device", "os", "channel", "sec", "is_attributed"])
    df["click_time"] = T0 + pd.to_timedelta(df.pop("sec"), unit="s")
    df["attributed_time"] = pd.NaT
    df = df.sort_values("click_time", kind="stable").reset_index(drop=True)
    return df[["ip", "app", "device", "os", "channel", "click_time", "attributed_time", "is_attributed"]]
