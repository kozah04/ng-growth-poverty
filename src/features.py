"""Feature engineering helpers for panel-style macroeconomic data."""

from collections.abc import Sequence

import pandas as pd


def interpolate_panel(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("economy", "series"),
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    """Linearly interpolate values within each panel group."""
    out = df.copy()
    out = out.sort_values([*group_cols, time_col]).reset_index(drop=True)
    out[value_col] = out.groupby(list(group_cols), sort=False)[value_col].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    return out


def add_differences(
    df: pd.DataFrame,
    periods: Sequence[int] = (1,),
    group_cols: Sequence[str] = ("economy", "series"),
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    """Add lagged difference columns for the selected periods."""
    out = df.copy()
    out = out.sort_values([*group_cols, time_col]).reset_index(drop=True)

    for p in periods:
        col = f"{value_col}_diff_{p}"
        out[col] = out.groupby(list(group_cols), sort=False)[value_col].transform(
            lambda s: s.diff(p)
        )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Default feature pipeline: interpolate missing values then add first diff."""
    out = interpolate_panel(df)
    out = add_differences(out, periods=(1,))
    return out
