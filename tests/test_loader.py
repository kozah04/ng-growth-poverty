import pandas as pd
import pytest

from src.loader import RAW_SCHEMA_COLUMNS, validate_raw_schema, world_bank_raw_to_long


def test_world_bank_raw_to_long_creates_canonical_schema():
    raw = pd.DataFrame(
        {
            "YR1990": [1.0, 2.0],
            "YR1991": [3.0, 4.0],
        },
        index=pd.MultiIndex.from_tuples(
            [("NGA", "NY.GDP.PCAP.KD"), ("GHA", "SI.POV.DDAY")],
            names=["economy", "series"],
        ),
    )

    long_df = world_bank_raw_to_long(raw)

    assert list(long_df.columns) == RAW_SCHEMA_COLUMNS
    assert len(long_df) == 4
    assert long_df.iloc[0].to_dict() == {
        "country": "GHA",
        "indicator": "SI.POV.DDAY",
        "year": 1990,
        "value": 2.0,
    }


def test_validate_raw_schema_rejects_duplicate_keys():
    df = pd.DataFrame(
        [
            {"country": "NGA", "indicator": "X", "year": 1990, "value": 1.0},
            {"country": "NGA", "indicator": "X", "year": 1990, "value": 2.0},
        ]
    )

    with pytest.raises(ValueError, match="duplicate panel keys"):
        validate_raw_schema(df)


def test_validate_raw_schema_rejects_missing_columns():
    df = pd.DataFrame({"country": ["NGA"], "year": [1990]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_schema(df)
