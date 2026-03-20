"""Model wrappers: OLS helpers, panel FE wrapper, and Granger utility."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: Sequence[str]):
    """Fit an OLS regression with intercept and return statsmodels results."""
    clean = df[[y_col, *x_cols]].dropna()
    y = clean[y_col]
    X = sm.add_constant(clean[list(x_cols)], has_constant="add")
    model = sm.OLS(y, X)
    return model.fit()


def fit_panel_fixed_effects(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    entity_col: str = "economy",
    time_col: str = "time",
):
    """Fit a simple entity fixed-effects panel model (requires linearmodels)."""
    try:
        from linearmodels.panel import PanelOLS
    except ImportError as exc:
        raise ImportError(
            "linearmodels is required for panel FE. Install it in environment.yml."
        ) from exc

    clean = df[[entity_col, time_col, y_col, *x_cols]].dropna().copy()
    clean = clean.set_index([entity_col, time_col])

    y = clean[y_col]
    X = sm.add_constant(clean[list(x_cols)], has_constant="add")
    model = PanelOLS(y, X, entity_effects=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def granger_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "economy",
    max_lag: int = 2,
) -> dict[str, dict[str, Any]]:
    """Run Granger causality tests per group and return raw test output."""
    outputs: dict[str, dict[str, Any]] = {}
    for group, group_df in df.groupby(group_col):
        pair = group_df[[y_col, x_col]].dropna()
        if len(pair) <= max_lag + 1:
            outputs[str(group)] = {"error": "insufficient_rows"}
            continue
        outputs[str(group)] = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
    return outputs
