"""
fetch_tea_staar.py
===================

This module contains helper routines for acquiring and preparing Texas STAAR
aggregate data from the Texas Education Agency (TEA).  The STAAR (State of
Texas Assessments of Academic Readiness) program publishes performance
statistics for each campus and district annually, including the percentage of
students meeting or exceeding proficiency benchmarks.  These data provide a
quantitative proxy for education quality that can be aggregated to a city level
for our housing appreciation model.

The TEA does not provide a simple public API for bulk downloads of STAAR
results, so this script implements two strategies:

1. **Attempt to download via known URLs.**  Some annual datasets may be hosted
   at stable links.  When a direct link is available for a given year, the
   script will download the CSV and save it to ``data/raw/tea_staar/``.

2. **Prompt the user for manual downloads.**  If a year cannot be
   automatically retrieved, the script prints detailed instructions for the
   researcher to follow.  Typically this involves visiting the TEA Research
   Portal, selecting the STAAR aggregate data for the desired year, and
   exporting district and campus CSVs.  Once downloaded, these files should
   be saved in ``data/raw/tea_staar/`` using the naming convention
   ``staar_{year}_district.csv`` and ``staar_{year}_campus.csv``.

Additionally, this module exposes a ``standardize_staar_columns`` function that
renames common STAAR fields to canonical names (e.g., ``city``, ``enrollment``,
``pct_meets``, ``pct_masters``) so that downstream scripts can process files
from different years uniformly.

Usage::

    python src/fetch_tea_staar.py --years 2018 2019 2020

The above example will attempt to download the 2018–2020 STAAR CSVs.  For years
that cannot be fetched automatically, it will print instructions for manual
retrieval.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


# Directory where raw STAAR files are stored
RAW_DIR: str = os.path.join("data", "raw", "tea_staar")

# Known download patterns for STAAR datasets.  Keys are years and values are
# URL templates.  If you find a stable pattern for STAAR downloads, add it
# here.  Leave the URL as None if no direct download is available for that
# year.
KNOWN_DOWNLOADS = {
    # Example: 2022: "https://tea.texas.gov/sites/default/files/2022_STAAR_Aggregate_Data.csv",
    # Entries will default to None for all unspecified years.
}


def download_or_prompt(years: Iterable[int]) -> None:
    """Attempt to download STAAR data for the given years, or prompt the user.

    For each year in ``years``, this function checks whether a CSV already
    exists in ``RAW_DIR``.  If not, it attempts to download the file from a
    known URL.  If the download pattern is unknown for that year, it prints
    instructions to the console describing how to manually export the data via
    the TEA Research Portal.

    Parameters
    ----------
    years : iterable of int
        The years of STAAR data to retrieve.
    """
    raw_dir = Path(RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        # Build expected filename
        dest = raw_dir / f"staar_{year}.csv"
        if dest.exists():
            print(f"STAAR data for {year} already present at {dest}")
            continue

        url_template = KNOWN_DOWNLOADS.get(year)
        if url_template:
            url = url_template.format(year=year)
            try:
                print(f"Attempting download of STAAR data for {year} from {url} ...")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    fh.write(resp.content)
                print(f"Downloaded STAAR {year} to {dest}")
                continue
            except Exception as exc:
                print(
                    f"Failed to download STAAR data for {year} from {url}: {exc}."
                    " Please download manually via the TEA Research Portal."
                )

        # Prompt for manual download
        print(
            f"No automated download configured for STAAR {year}.\n"
            f"Please visit the TEA Research Portal and export the STAAR aggregate\n"
            f"district and campus CSVs for {year}.  Save them as:\n"
            f"  {RAW_DIR}/staar_{year}_district.csv\n"
            f"  {RAW_DIR}/staar_{year}_campus.csv\n"
            "Once saved, rerun the pipeline.\n"
        )


def standardize_staar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names in a STAAR dataset.

    Different STAAR exports contain slightly different column names for
    equivalent concepts.  This helper renames the following fields when
    present:

    - city -> ``city`` (case insensitive)
    - total_enrollment or enrollment -> ``enrollment``
    - % Meets Grade Level or better -> ``pct_meets``
    - % Masters Grade Level -> ``pct_masters``

    Parameters
    ----------
    df : pandas.DataFrame
        A STAAR dataset loaded from CSV.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with renamed columns.  Columns that are not recognized
        are left unchanged.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        lower = col.lower().strip()
        # City
        if lower == "city":
            rename_map[col] = "city"
        # Enrollment
        elif lower in {"enrollment", "total_enrollment", "total students"}:
            rename_map[col] = "enrollment"
        # Meets grade level percentage
        elif re.match(r"%? ?(students )?meets", lower) or "meets grade level" in lower:
            rename_map[col] = "pct_meets"
        # Masters grade level percentage
        elif "masters grade level" in lower or re.match(r"%? ?(students )?masters", lower):
            rename_map[col] = "pct_masters"

    return df.rename(columns=rename_map, inplace=False)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for this module."""
    parser = argparse.ArgumentParser(description="Download or prepare STAAR data")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[],
        help="Years of STAAR data to download or prepare (e.g. 2018 2019 2020)",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    ns = parse_args()
    if not ns.years:
        print(
            "No years specified. Pass one or more years with --years to download or "
            "prepare STAAR data."
        )
        sys.exit(0)
    download_or_prompt(ns.years)