"""Evaluation helpers: diagnostics and report export."""

from pathlib import Path
from typing import Any

import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera


def regression_diagnostics(fitted_model) -> dict[str, Any]:
    """Return a compact set of model diagnostics for quick reporting."""
    resid = fitted_model.resid
    exog = fitted_model.model.exog

    jb_stat, jb_p, _, _ = jarque_bera(resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)

    return {
        "nobs": float(fitted_model.nobs),
        "r2": float(getattr(fitted_model, "rsquared", float("nan"))),
        "adj_r2": float(getattr(fitted_model, "rsquared_adj", float("nan"))),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_pvalue": float(bp_p),
    }


def save_regression_table(table_df: pd.DataFrame, output_csv: str | Path) -> Path:
    """Save a regression table as CSV under outputs/reports."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False)
    return output_path
