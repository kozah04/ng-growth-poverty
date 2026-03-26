import shutil
import uuid
from pathlib import Path

import pandas as pd

from pipeline import run_features


class Args:
    def __init__(self, input_parquet: str, output_parquet: str):
        self.input_parquet = input_parquet
        self.output_parquet = output_parquet


def test_run_features_smoke(tmp_path):
    raw = pd.DataFrame(
        [
            {"country": "NGA", "indicator": "NY.GDP.PCAP.KD", "year": 1990, "value": 100.0},
            {"country": "NGA", "indicator": "NY.GDP.PCAP.KD", "year": 1991, "value": 110.0},
            {"country": "NGA", "indicator": "NY.GDP.MKTP.KD.ZG", "year": 1990, "value": 2.0},
            {"country": "NGA", "indicator": "NY.GDP.MKTP.KD.ZG", "year": 1991, "value": 3.0},
            {"country": "NGA", "indicator": "DT.ODA.ALLD.KD", "year": 1990, "value": 5.0},
            {"country": "NGA", "indicator": "DT.ODA.ALLD.KD", "year": 1991, "value": 6.0},
            {"country": "NGA", "indicator": "FP.CPI.TOTL.ZG", "year": 1990, "value": 7.0},
            {"country": "NGA", "indicator": "FP.CPI.TOTL.ZG", "year": 1991, "value": 8.0},
            {"country": "NGA", "indicator": "SL.UEM.TOTL.ZS", "year": 1990, "value": 9.0},
            {"country": "NGA", "indicator": "SL.UEM.TOTL.ZS", "year": 1991, "value": 10.0},
            {"country": "NGA", "indicator": "SI.POV.DDAY", "year": 1990, "value": 50.0},
            {"country": "NGA", "indicator": "SI.POV.DDAY", "year": 1991, "value": 45.0},
            {"country": "NGA", "indicator": "SI.POV.GINI", "year": 1990, "value": 40.0},
            {"country": "NGA", "indicator": "SI.POV.GINI", "year": 1991, "value": 39.0},
        ]
    )

    input_path = tmp_path / "wb_raw.parquet"
    output_path = tmp_path / "panel.parquet"
    raw.to_parquet(input_path, index=False)

    run_features(Args(str(input_path), str(output_path)))

    out = pd.read_parquet(output_path)
    assert list(out["country"].unique()) == ["NGA"]
    assert "poverty" in out.columns
