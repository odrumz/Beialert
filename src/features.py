"""Turn one monthly price series into a supervised learning table.

Every rolling statistic is shifted by one month. Without that shift the
rolling mean would include the month being predicted, which leaks the target
into its own feature and produces scores that collapse in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(monthly: pd.Series) -> pd.DataFrame:
    """Return a frame indexed by month with FEATURES + TARGET columns."""
    frame = monthly.to_frame(name=config.TARGET).copy()
    past = frame[config.TARGET].shift(1)  # everything is built from the past only

    for n in config.LAGS:
        frame[f"lag_{n}"] = frame[config.TARGET].shift(n)

    for w in config.ROLL_WINDOWS:
        frame[f"roll_mean_{w}"] = past.rolling(w, min_periods=w).mean()
        frame[f"roll_std_{w}"] = past.rolling(w, min_periods=w).std()

    frame["mom_change"] = frame["lag_1"] / frame["lag_2"] - 1.0

    # Cyclical month: December sits next to January instead of 11 units away.
    month = frame.index.month
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)

    return frame


def model_frame(monthly: pd.Series) -> pd.DataFrame:
    """Feature frame with rows that cannot be used dropped."""
    frame = build_features(monthly)
    return frame.dropna(subset=config.FEATURES + [config.TARGET])


def row_from_history(history: pd.Series, target_month: pd.Timestamp) -> pd.DataFrame:
    """Build a single feature row for a month we do not have a price for.

    `history` is a month-indexed series of known or already-predicted prices,
    ending the month before `target_month`. Used by the recursive forecaster.
    """
    history = history.sort_index()
    values = {}

    for n in config.LAGS:
        stamp = target_month - pd.DateOffset(months=n)
        values[f"lag_{n}"] = history.get(stamp, np.nan)

    prior = history[history.index < target_month]
    for w in config.ROLL_WINDOWS:
        window = prior.tail(w)
        values[f"roll_mean_{w}"] = window.mean() if len(window) == w else np.nan
        values[f"roll_std_{w}"] = window.std() if len(window) == w else np.nan

    values["mom_change"] = (
        values["lag_1"] / values["lag_2"] - 1.0
        if values.get("lag_2") not in (None, 0) and not pd.isna(values.get("lag_2"))
        else np.nan
    )
    values["month_sin"] = np.sin(2 * np.pi * target_month.month / 12)
    values["month_cos"] = np.cos(2 * np.pi * target_month.month / 12)

    return pd.DataFrame([values], index=[target_month])[config.FEATURES]
