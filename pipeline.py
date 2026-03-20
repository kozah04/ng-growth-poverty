"""Simple CLI pipeline entrypoint for pull and feature steps."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.features import build_features
from src.loader import wb_pull_and_cache


DEFAULT_INDICATORS = [
    "NY.GDP.PCAP.KD",
    "NY.GDP.MKTP.KD.ZG",
    "SI.POV.DDAY",
    "SI.POV.GINI",
    "NE.TRD.GNFS.ZS",
    "DT.ODA.ALLD.KD",
    "FP.CPI.TOTL.ZG",
    "SL.UEM.TOTL.ZS",
]

DEFAULT_COUNTRIES = ["NGA", "GHA", "KEN", "ZAF", "ETH"]


def run_pull(args) -> None:
    _, cache_path = wb_pull_and_cache(
        indicators=DEFAULT_INDICATORS,
        countries=DEFAULT_COUNTRIES,
        start_year=args.start_year,
        end_year=args.end_year,
        cache_name=args.cache_name,
    )
    print(f"Saved raw pull to: {cache_path}")


def run_features(args) -> None:
    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet)

    df = pd.read_parquet(input_path)
    featured = build_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(output_path, index=False)
    print(f"Saved processed features to: {output_path}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ng-growth-poverty pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="Pull data from World Bank API and cache raw data")
    pull.add_argument("--start-year", type=int, default=1990)
    pull.add_argument("--end-year", type=int, default=2023)
    pull.add_argument("--cache-name", default="world_bank_pull.parquet")
    pull.set_defaults(func=run_pull)

    feats = sub.add_parser("features", help="Run feature engineering from raw parquet")
    feats.add_argument(
        "--input-parquet",
        default="data/raw/world_bank_pull.parquet",
    )
    feats.add_argument(
        "--output-parquet",
        default="data/processed/features.parquet",
    )
    feats.set_defaults(func=run_features)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
