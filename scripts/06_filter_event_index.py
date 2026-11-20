"""
Write a subset of an events index for an inclusive year range, keeping the first N fires per year
in the input CSV's current order.

    python scripts/06_filter_event_index.py --index data/events_index.csv \
        --out data/events_index_subset.csv --start-year 2018 --end-year 2021 \
        --first-n 10 --year-stride
"""
import argparse
from pathlib import Path

import pandas as pd


def filter_events(events, start_year, end_year, first_n):
    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year")
    if first_n <= 0:
        raise ValueError("Number of fires per year must be positive")
    if "year" not in events.columns:
        raise ValueError("Events index must contain a 'year' column")

    years = pd.to_numeric(events["year"], errors="raise")
    if years.isna().any():
        raise ValueError("Events index contains missing years")
    in_range = years.between(start_year, end_year)
    selected = events.loc[in_range]
    return selected.groupby(years.loc[in_range].to_numpy(), sort=False).head(first_n)


def interleave_years(events):
    """Round-robin rows by ascending year, retaining input order within each year."""
    year_values = pd.to_numeric(events["year"], errors="raise").to_numpy()
    positions_by_year = {}
    for position, year in enumerate(year_values):
        positions_by_year.setdefault(year, []).append(position)

    ordered_positions = []
    years = sorted(positions_by_year)
    for offset in range(max((len(positions) for positions in positions_by_year.values()), default=0)):
        for year in years:
            if offset < len(positions_by_year[year]):
                ordered_positions.append(positions_by_year[year][offset])
    return events.iloc[ordered_positions]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Keep the first N fires per year in an inclusive year range."
    )
    parser.add_argument("--index", default="data/events_index.csv", help="source events CSV")
    parser.add_argument("--out", default="data/events_index_subset.csv", help="filtered output CSV")
    parser.add_argument("--start-year", type=int, required=True, help="first year to include")
    parser.add_argument("--end-year", type=int, required=True, help="last year to include")
    parser.add_argument("--first-n", type=int, required=True, help="fires to keep per year")
    parser.add_argument(
        "--year-stride",
        action="store_true",
        help="interleave rows round-robin in ascending year order",
    )
    args = parser.parse_args(argv)

    source = Path(args.index)
    output = Path(args.out)
    if source.resolve() == output.resolve():
        parser.error("Output path must differ from the input index")

    try:
        events = pd.read_csv(source)
        selected = filter_events(events, args.start_year, args.end_year, args.first_n)
        if args.year_stride:
            selected = interleave_years(selected)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    counts = selected.groupby("year", sort=False).size().to_dict()
    print(f"Wrote {len(selected)} fires to {output}")
    print(f"Fires per year: {counts}")


if __name__ == "__main__":
    main()
