import pandas as pd
import pytest

from src.models import fit_ols, granger_by_group


# ---------------------------------------------------------------------------
# fit_ols
# ---------------------------------------------------------------------------


def test_fit_ols_returns_significant_coefficient():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5], "x": [2, 4, 6, 8, 10]})
    result = fit_ols(df, y_col="y", x_cols=["x"])

    assert result.rsquared == pytest.approx(1.0)
    assert result.pvalues["x"] < 0.05


def test_fit_ols_drops_nan_rows():
    df = pd.DataFrame({"y": [1, None, 3, 4, 5], "x": [2, 4, 6, 8, 10]})
    result = fit_ols(df, y_col="y", x_cols=["x"])

    assert result.nobs == 4


# ---------------------------------------------------------------------------
# granger_by_group
# ---------------------------------------------------------------------------


def _granger_panel() -> pd.DataFrame:
    """Create a minimal panel with two countries and enough rows for lag=2."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for country in ["A", "B"]:
        for year in range(2000, 2020):
            rows.append({
                "country": country,
                "year": year,
                "x": float(year - 2000) + rng.normal(0, 0.5),
                "y": float((year - 2000) * 2) + rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


def test_granger_by_group_returns_results_for_each_country():
    df = _granger_panel()
    results = granger_by_group(df, x_col="x", y_col="y", max_lag=2)

    assert set(results.keys()) == {"A", "B"}
    # Each country should have lag keys 1 and 2
    for group in results.values():
        assert 1 in group
        assert 2 in group


def test_granger_by_group_insufficient_rows():
    df = pd.DataFrame({
        "country": ["A", "A"],
        "year": [2000, 2001],
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
    })
    results = granger_by_group(df, x_col="x", y_col="y", max_lag=2)

    assert results["A"]["error"] == "insufficient_rows"


def test_granger_by_group_constant_series():
    df = pd.DataFrame({
        "country": ["A"] * 10,
        "year": list(range(2000, 2010)),
        "x": [5.0] * 10,
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })
    results = granger_by_group(df, x_col="x", y_col="y", max_lag=2)

    assert results["A"]["error"] == "constant_series"


def test_granger_by_group_duplicate_time():
    df = pd.DataFrame({
        "country": ["A"] * 10,
        "year": [2000, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008],
        "x": list(range(10, 20)),
        "y": list(range(20, 30)),
    })
    results = granger_by_group(df, x_col="x", y_col="y", max_lag=2)

    assert results["A"]["error"] == "duplicate_time_values"
