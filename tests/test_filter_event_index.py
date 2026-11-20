import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_filter_event_index.py"
spec = importlib.util.spec_from_file_location("filter_event_index", SCRIPT)
filter_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_script)


def test_filter_keeps_first_n_per_year_in_input_order():
    events = pd.DataFrame(
        {
            "year": [2019, 2020, 2019, 2021, 2020, 2019, 2020, 2022],
            "fire": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )

    selected = filter_script.filter_events(events, 2019, 2021, 2)

    assert selected["fire"].tolist() == ["a", "b", "c", "d", "e"]


def test_year_stride_round_robins_and_skips_exhausted_years():
    events = pd.DataFrame(
        {
            "year": [2020, 2018, 2020, 2021, 2018, 2019, 2020, 2021, 2019],
            "fire": ["20a", "18a", "20b", "21a", "18b", "19a", "20c", "21b", "19b"],
        }
    )

    selected = filter_script.filter_events(events, 2018, 2021, 3)
    interleaved = filter_script.interleave_years(selected)

    assert interleaved["fire"].tolist() == [
        "18a", "19a", "20a", "21a",
        "18b", "19b", "20b", "21b",
        "20c",
    ]


def test_cli_writes_filtered_csv(tmp_path):
    source = tmp_path / "events.csv"
    output = tmp_path / "filtered" / "events.csv"
    pd.DataFrame(
        {"year": [2018, 2020, 2019, 2019], "fire": ["a", "d", "b", "c"]}
    ).to_csv(source, index=False)

    filter_script.main(
        [
            "--index", str(source),
            "--out", str(output),
            "--start-year", "2019",
            "--end-year", "2020",
            "--first-n", "1",
            "--year-stride",
        ]
    )

    filtered = pd.read_csv(output)
    assert filtered["fire"].tolist() == ["b", "d"]


@pytest.mark.parametrize(
    ("start_year", "end_year", "first_n", "message"),
    [
        (2020, 2019, 1, "Start year"),
        (2019, 2020, 0, "positive"),
    ],
)
def test_filter_rejects_invalid_range_or_limit(start_year, end_year, first_n, message):
    events = pd.DataFrame({"year": [2019], "fire": ["a"]})

    with pytest.raises(ValueError, match=message):
        filter_script.filter_events(events, start_year, end_year, first_n)
