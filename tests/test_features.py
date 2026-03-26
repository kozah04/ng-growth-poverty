import pandas as pd

from src.features import build_model_panel, interpolate_panel


def _sample_raw() -> pd.DataFrame:
    rows = []
    for country in ["NGA", "GHA"]:
        for year in [1990, 1991, 1992]:
            rows.extend(
                [
                    {
                        "country": country,
                        "indicator": "NY.GDP.PCAP.KD",
                        "year": year,
                        "value": 100.0 + year if country == "NGA" else 200.0 + year,
                    },
                    {
                        "country": country,
                        "indicator": "NY.GDP.MKTP.KD.ZG",
                        "year": year,
                        "value": float(year - 1989),
                    },
                    {
                        "country": country,
                        "indicator": "DT.ODA.ALLD.KD",
                        "year": year,
                        "value": 10.0 * (year - 1989),
                    },
                    {
                        "country": country,
                        "indicator": "FP.CPI.TOTL.ZG",
                        "year": year,
                        "value": 5.0,
                    },
                    {
                        "country": country,
                        "indicator": "SL.UEM.TOTL.ZS",
                        "year": year,
                        "value": 7.0,
                    },
                ]
            )

        rows.extend(
            [
                {
                    "country": country,
                    "indicator": "SI.POV.DDAY",
                    "year": 1990,
                    "value": 50.0 if country == "NGA" else 40.0,
                },
                {
                    "country": country,
                    "indicator": "SI.POV.DDAY",
                    "year": 1991,
                    "value": None,
                },
                {
                    "country": country,
                    "indicator": "SI.POV.DDAY",
                    "year": 1992,
                    "value": 30.0 if country == "NGA" else 20.0,
                },
                {
                    "country": country,
                    "indicator": "SI.POV.GINI",
                    "year": 1990,
                    "value": 45.0,
                },
                {
                    "country": country,
                    "indicator": "SI.POV.GINI",
                    "year": 1991,
                    "value": None,
                },
                {
                    "country": country,
                    "indicator": "SI.POV.GINI",
                    "year": 1992,
                    "value": 35.0,
                },
            ]
        )

    return pd.DataFrame(rows)


def test_interpolate_panel_stays_within_observed_bounds():
    raw = pd.DataFrame(
        [
            {"country": "NGA", "indicator": "SI.POV.DDAY", "year": 1990, "value": None},
            {"country": "NGA", "indicator": "SI.POV.DDAY", "year": 1991, "value": 10.0},
            {"country": "NGA", "indicator": "SI.POV.DDAY", "year": 1992, "value": None},
            {"country": "NGA", "indicator": "SI.POV.DDAY", "year": 1993, "value": 30.0},
            {"country": "NGA", "indicator": "SI.POV.DDAY", "year": 1994, "value": None},
        ]
    )

    out = interpolate_panel(raw)
    values = out["value"].tolist()

    assert pd.isna(values[0])
    assert values[1:4] == [10.0, 20.0, 30.0]
    assert pd.isna(values[4])


def test_build_model_panel_creates_expected_columns_and_observed_flags():
    panel = build_model_panel(_sample_raw(), lags=(1,))

    assert {"country", "year", "poverty", "poverty_observed", "log_gdp_pc_lag1"} <= set(
        panel.columns
    )

    nga_1991 = panel[(panel["country"] == "NGA") & (panel["year"] == 1991)].iloc[0]
    assert nga_1991["poverty"] == 40.0
    assert pd.isna(nga_1991["poverty_observed"])

    nga_1990 = panel[(panel["country"] == "NGA") & (panel["year"] == 1990)].iloc[0]
    assert nga_1990["poverty_observed"] == 50.0
