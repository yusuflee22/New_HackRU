"""
join_zillow_education.py
=======================

This script prepares a long‑format panel of Texas Zillow price data that is
independent of any education metrics.  The filtered Zillow dataset contains
monthly ZHVI observations for each city in wide format.  This script melts
those columns into a long table where each row corresponds to a single
city–month observation and adds a normalised city name column.  The
resulting panel is written to ``data/processed/panel_tx_city_year.csv`` and
serves as input for the feature snapshot and modelling stages when no
education data are used.

Usage::

    python src/join_zillow_education.py

Prerequisites:

1. Run ``filter_zillow_tx_metros.py`` to produce ``data/processed/zhvi_tx_metros.csv``.

If you later obtain education data, you can adjust this script to perform a
join on ``city_norm`` and ``year`` as originally intended.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

try:
    # Prefer to reuse the normalisation function from the education module if available
    from .build_city_year_education import normalize_city  # type: ignore[attr-defined]
except Exception:
    # Fallback normalisation: upper‑case, remove punctuation, collapse spaces
    import re

    def normalize_city(name: str) -> str:
        if not isinstance(name, str) or not name:
            return ""
        s = name.upper()
        s = re.sub(r"[^A-Z0-9 ]", "", s)
        s = " ".join(s.split())
        return s


# Input path
ZHVI_PATH: str = os.path.join("data", "processed", "zhvi_tx_metros.csv")

# Output path
OUT_PATH: str = os.path.join("data", "processed", "panel_tx_city_year.csv")


def melt_zillow(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the wide Zillow dataset into long format with year.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with city‐level ZHVI in wide format.  The columns should
        include metadata columns (e.g., ``RegionName``, ``Metro``, ``State``)
        and a series of monthly date columns (``YYYY-MM``).

    Returns
    -------
    pandas.DataFrame
        Long‑format DataFrame with columns: ``RegionName``, ``Metro``,
        ``State``, ``date``, ``year``, and ``zhvi``.
    """
    # Identify date columns by pattern
    date_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(c))]
    if not date_cols:
        date_cols = [c for c in df.columns if re.match(r"\d{4}-\d{2}", str(c))]
    meta_cols = [c for c in df.columns if c not in date_cols]
    # Melt to long format
    long_df = df.melt(id_vars=meta_cols, value_vars=date_cols, var_name="date", value_name="zhvi")
    # Extract year as integer from date
    long_df["year"] = long_df["date"].str[:4].astype(int)
    return long_df


def create_panel(
    zillow_path: str = ZHVI_PATH,
    out_path: str = OUT_PATH,
) -> None:
    """Create a long‑format panel from the Zillow data without education.

    Parameters
    ----------
    zillow_path : str
        Path to the filtered Zillow CSV produced by ``filter_zillow_tx_metros.py``.
    out_path : str
        Destination for the panel CSV.
    """
    zillow_file = Path(zillow_path)
    if not zillow_file.exists():
        raise FileNotFoundError(
            "The Zillow input file was not found. Run filter_zillow_tx_metros.py first."
        )
    zillow_df = pd.read_csv(zillow_file)
    zillow_long = melt_zillow(zillow_df)
    # Normalise city names
    zillow_long["city_norm"] = zillow_long["RegionName"].apply(normalize_city)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    zillow_long.to_csv(out_file, index=False)
    print(f"Panel written to {out_file} ({len(zillow_long)} rows)")


if __name__ == "__main__":
    try:
        create_panel()
    except Exception as exc:
        print(f"Failed to create panel: {exc}", file=sys.stderr)
        sys.exit(1)