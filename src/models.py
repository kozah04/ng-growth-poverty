"""Model wrappers: OLS helpers, panel FE wrapper, and safer Granger utility."""

from contextlib import redirect_stdout
import io
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
    entity_col: str = "country",
    time_col: str = "year",
):
    """Fit an entity fixed-effects panel model with clustered standard errors."""
    try:
        from linearmodels.panel import PanelOLS
    except ImportError as exc:
        raise ImportError(
            "linearmodels is required for panel FE. Install it in environment.yml."
        ) from exc

    clean = df[[entity_col, time_col, y_col, *x_cols]].dropna().copy()
    clean = clean.set_index([entity_col, time_col]).sort_index()

    y = clean[y_col]
    X = clean[list(x_cols)]
    model = PanelOLS(y, X, entity_effects=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def granger_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "country",
    time_col: str = "year",
    max_lag: int = 2,
    min_rows: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run Granger causality tests per group with ordering and validity checks."""
    required_rows = min_rows if min_rows is not None else max_lag + 2
    outputs: dict[str, dict[str, Any]] = {}

    for group, group_df in df.groupby(group_col):
        ordered = group_df.sort_values(time_col)
        years = ordered[time_col]

        if years.duplicated().any():
            outputs[str(group)] = {"error": "duplicate_time_values"}
            continue

        pair = ordered[[y_col, x_col]].dropna()
        if len(pair) < required_rows:
            outputs[str(group)] = {
                "error": "insufficient_rows",
                "n_rows": int(len(pair)),
                "required_rows": int(required_rows),
            }
            continue

        if pair.nunique().min() <= 1:
            outputs[str(group)] = {"error": "constant_series"}
            continue

        try:
            with redirect_stdout(io.StringIO()):
                outputs[str(group)] = grangercausalitytests(pair, maxlag=max_lag)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            outputs[str(group)] = {"error": type(exc).__name__, "message": str(exc)}


    return outputs
