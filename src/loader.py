"""World Bank data pull utilities with a canonical long-form raw schema."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

RAW_SCHEMA_COLUMNS = ["country", "indicator", "year", "value"]


def _normalize_wb_year(label: object) -> int:
    text = str(label)
    return int(text.replace("YR", ""))


def world_bank_raw_to_long(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a World Bank MultiIndex dataframe to the repo's canonical raw schema."""
    if raw_df.index.nlevels != 2 or raw_df.index.names != ["economy", "series"]:
        raise ValueError(
            "Expected a World Bank dataframe indexed by ['economy', 'series']."
        )

    normalized = raw_df.rename_axis(index=["country", "indicator"]).rename(
        columns=_normalize_wb_year
    )
    try:
        stacked = normalized.stack(future_stack=True)
    except TypeError:
        stacked = normalized.stack(dropna=False)

    long_df = (
        stacked.rename("value")
        .reset_index()
        .rename(columns={"level_2": "year"})
    )

    long_df["year"] = long_df["year"].astype(int)
    long_df = long_df.sort_values(["country", "indicator", "year"]).reset_index(
        drop=True
    )
    return long_df[RAW_SCHEMA_COLUMNS]


def validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical raw pull schema."""
    missing = [col for col in RAW_SCHEMA_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing required columns {missing}. "
            f"Expected columns: {RAW_SCHEMA_COLUMNS}."
        )

    out = df.loc[:, RAW_SCHEMA_COLUMNS].copy()
    out["country"] = out["country"].astype(str)
    out["indicator"] = out["indicator"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    duplicate_mask = out.duplicated(["country", "indicator", "year"])
    if duplicate_mask.any():
        duplicates = out.loc[duplicate_mask, ["country", "indicator", "year"]]
        sample = duplicates.head(5).to_dict(orient="records")
        raise ValueError(f"Raw data contains duplicate panel keys, sample={sample}")

    return out.sort_values(["country", "indicator", "year"]).reset_index(drop=True)


def wb_pull_and_cache(
    indicators: Iterable[str],
    countries: Iterable[str],
    start_year: int,
    end_year: int,
    cache_name: str = "wb_raw.parquet",
) -> tuple[pd.DataFrame, Path]:
    """Pull panel data from the World Bank API and cache it in canonical long form."""
    import wbgapi as wb

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / cache_name

    raw_df = wb.data.DataFrame(
        list(indicators),
        economy=list(countries),
        time=range(start_year, end_year + 1),
    )
    long_df = validate_raw_schema(world_bank_raw_to_long(raw_df))

    long_df.to_parquet(cache_path, index=False)
    return long_df, cache_path
