"""CLI entrypoint for pulling raw data and building the modelling panel."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_COUNTRIES,
    DEFAULT_END_YEAR,
    DEFAULT_INDICATORS,
    DEFAULT_PANEL_PATH,
    DEFAULT_RAW_PATH,
    DEFAULT_START_YEAR,
)
from src.features import build_features
from src.loader import wb_pull_and_cache


def run_pull(args) -> None:
    raw_df, cache_path = wb_pull_and_cache(
        indicators=DEFAULT_INDICATORS,
        countries=DEFAULT_COUNTRIES,
        start_year=args.start_year,
        end_year=args.end_year,
        cache_name=args.cache_name,
    )
    print(f"Saved raw pull to: {cache_path} ({len(raw_df)} rows)")


def run_features(args) -> None:
    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    featured = build_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(output_path, index=False)
    print(f"Saved processed panel to: {output_path}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ng-growth-poverty pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="Pull data from World Bank API and cache raw data")
    pull.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    pull.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    pull.add_argument("--cache-name", default=DEFAULT_RAW_PATH.name)
    pull.set_defaults(func=run_pull)

    feats = sub.add_parser(
        "features", help="Build the processed country-year panel from raw parquet"
    )
    feats.add_argument(
        "--input-parquet",
        default=str(DEFAULT_RAW_PATH),
    )
    feats.add_argument(
        "--output-parquet",
        default=str(DEFAULT_PANEL_PATH),
    )
    feats.set_defaults(func=run_features)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
