"""World Bank data pull utilities with local parquet caching."""

from pathlib import Path
from typing import Iterable

import pandas as pd
import wbgapi as wb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def wb_pull_and_cache(
    indicators: Iterable[str],
    countries: Iterable[str],
    start_year: int,
    end_year: int,
    cache_name: str = "world_bank_pull.parquet",
) -> tuple[pd.DataFrame, Path]:
    """Pull panel data from World Bank and cache the raw pull as parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / cache_name

    df = wb.data.DataFrame(
        list(indicators),
        economy=list(countries),
        time=range(start_year, end_year + 1),
    ).reset_index()

    df.to_parquet(cache_path, index=False)
    return df, cache_path
