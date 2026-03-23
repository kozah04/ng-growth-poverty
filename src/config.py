"""Project-wide constants: indicators, countries, paths, and defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

DEFAULT_RAW_PATH = RAW_DIR / "wb_raw.parquet"
DEFAULT_PANEL_PATH = PROCESSED_DIR / "panel.parquet"

# World Bank indicator codes mapped to short feature names.
# The keys are the human-readable names used in the processed panel;
# the values are the World Bank series codes passed to wbgapi.
DEFAULT_SERIES_MAP: dict[str, str] = {
    "gdp_pc": "NY.GDP.PCAP.KD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "oda": "DT.ODA.ALLD.KD",
    "inflation": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
    "poverty": "SI.POV.DDAY",
    "gini": "SI.POV.GINI",
}

# Trade is pulled for coverage analysis but not used in the modelling panel.
EXTRA_PULL_INDICATORS: list[str] = ["NE.TRD.GNFS.ZS"]

# All indicator codes to pull from the World Bank API.
DEFAULT_INDICATORS: list[str] = [
    *DEFAULT_SERIES_MAP.values(),
    *EXTRA_PULL_INDICATORS,
]

DEFAULT_COUNTRIES: list[str] = ["NGA", "GHA", "KEN", "ZAF", "ETH"]

COUNTRY_LABELS: dict[str, str] = {
    "NGA": "Nigeria",
    "GHA": "Ghana",
    "KEN": "Kenya",
    "ZAF": "South Africa",
    "ETH": "Ethiopia",
}

DEFAULT_START_YEAR = 1990
DEFAULT_END_YEAR = 2023

# Feature engineering defaults.
INTERPOLATED_COLUMNS = ("poverty", "gini")
DEFAULT_LAG_SOURCE = "log_gdp_pc"
DEFAULT_LAGS = (1, 2, 3)
