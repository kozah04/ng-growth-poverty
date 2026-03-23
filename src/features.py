"""Feature engineering helpers for World Bank panel data."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_LAG_SOURCE,
    DEFAULT_LAGS,
    DEFAULT_SERIES_MAP,
    INTERPOLATED_COLUMNS,
)
from src.loader import validate_raw_schema


def build_indicator_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the notebook-friendly indicator matrix indexed by country and indicator."""
    validated = validate_raw_schema(raw_df)
    matrix = (
        validated.assign(year_label=validated["year"].map(lambda year: f"YR{year}"))
        .pivot(index=["country", "indicator"], columns="year_label", values="value")
        .sort_index()
    )
    matrix.index = matrix.index.set_names(["economy", "series"])
    return matrix


def interpolate_panel(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("country", "indicator"),
    time_col: str = "year",
    value_col: str = "value",
) -> pd.DataFrame:
    """Linearly interpolate values within each panel group and within observed bounds."""
    out = df.sort_values([*group_cols, time_col]).copy()
    out[value_col] = (
        out.groupby(list(group_cols), sort=False)[value_col]
        .apply(lambda s: s.interpolate(method="linear", limit_area="inside"))
        .droplevel(list(range(len(list(group_cols)))))
    )
    return out.reset_index(drop=True)


def add_differences(
    df: pd.DataFrame,
    periods: Sequence[int] = (1,),
    group_cols: Sequence[str] = ("country", "indicator"),
    time_col: str = "year",
    value_col: str = "value",
) -> pd.DataFrame:
    """Add lagged difference columns for the selected periods."""
    out = df.sort_values([*group_cols, time_col]).copy()

    for period in periods:
        col = f"{value_col}_diff_{period}"
        out[col] = out.groupby(list(group_cols), sort=False)[value_col].diff(period)
    return out.reset_index(drop=True)


def _pivot_raw_to_panel(
    raw_df: pd.DataFrame,
    series_map: Mapping[str, str],
) -> pd.DataFrame:
    subset = raw_df[raw_df["indicator"].isin(series_map.values())].copy()
    renamed = subset.assign(
        feature=subset["indicator"].map({v: k for k, v in series_map.items()})
    )

    panel = (
        renamed.pivot(index=["country", "year"], columns="feature", values="value")
        .reset_index()
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    return panel


def build_model_panel(
    raw_df: pd.DataFrame,
    series_map: Mapping[str, str] = DEFAULT_SERIES_MAP,
    interpolate_columns: Sequence[str] = INTERPOLATED_COLUMNS,
    lag_source_col: str = DEFAULT_LAG_SOURCE,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Build the canonical country-year panel used by the modelling workflow."""
    validated = validate_raw_schema(raw_df)
    transformed = validated.copy()

    observed_col_names: dict[str, str] = {}
    for feature_name in interpolate_columns:
        indicator = series_map[feature_name]
        mask = transformed["indicator"].eq(indicator)
        observed_col = f"{feature_name}_observed"
        transformed.loc[mask, observed_col] = transformed.loc[mask, "value"]
        observed_col_names[feature_name] = observed_col

    transformed = interpolate_panel(transformed)
    transformed = add_differences(transformed, periods=(1,))

    panel = _pivot_raw_to_panel(transformed, series_map=series_map)

    for feature_name, observed_col in observed_col_names.items():
        observed_values = (
            transformed.loc[
                transformed["indicator"].eq(series_map[feature_name]),
                ["country", "year", observed_col],
            ]
            .drop_duplicates(["country", "year"])
            .rename(columns={observed_col: f"{feature_name}_observed"})
        )
        panel = panel.merge(observed_values, on=["country", "year"], how="left")

    if "gdp_pc" in panel:
        panel["log_gdp_pc"] = np.log(panel["gdp_pc"].replace(0, np.nan))
    if "oda" in panel:
        panel["log_oda"] = np.log(panel["oda"].replace(0, np.nan))

    if lag_source_col in panel:
        for lag in lags:
            panel[f"{lag_source_col}_lag{lag}"] = panel.groupby("country")[
                lag_source_col
            ].shift(lag)

    return panel.sort_values(["country", "year"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the repository's canonical processed panel."""
    return build_model_panel(df)
