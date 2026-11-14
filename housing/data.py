"""Loading and preparing the California Housing data (1990 U.S. Census)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "housing.csv"

CATEGORIES = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
VALUE_CAP = 500_001

LABELS = {
    "median_house_value": "Median house value ($)",
    "median_income": "Median income ($10,000s)",
    "housing_median_age": "Median house age (years)",
    "population": "Population",
    "households": "Households",
    "total_rooms": "Total rooms",
    "total_bedrooms": "Total bedrooms",
    "rooms_per_household": "Rooms per household",
    "bedrooms_per_room": "Bedrooms per room",
    "people_per_household": "People per household",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "ocean_proximity": "Ocean proximity",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and add a few easy to read columns."""
    df = pd.read_csv(path)
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["people_per_household"] = df["population"] / df["households"]
    df["value_capped"] = df["median_house_value"] >= VALUE_CAP
    df["ocean_proximity"] = pd.Categorical(df["ocean_proximity"], categories=CATEGORIES)
    return df


def filter_data(df: pd.DataFrame, categories=None, income=None, age=None, value=None,
                include_capped: bool = True) -> pd.DataFrame:
    """Apply the dashboard filters. Each range is a (low, high) tuple or None."""
    m = np.ones(len(df), dtype=bool)
    if categories:
        m &= df["ocean_proximity"].isin(categories).to_numpy()
    for col, rng in (("median_income", income), ("housing_median_age", age),
                     ("median_house_value", value)):
        if rng is not None:
            m &= df[col].between(rng[0], rng[1]).to_numpy()
    if not include_capped:
        m &= ~df["value_capped"].to_numpy()
    return df[m]


def summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the filtered data."""
    if df.empty:
        return {"districts": 0}
    return {
        "districts": int(len(df)),
        "median_value": float(df["median_house_value"].median()),
        "mean_value": float(df["median_house_value"].mean()),
        "median_income": float(df["median_income"].median()),
        "population": int(df["population"].sum()),
        "capped_share": float(df["value_capped"].mean()),
        "missing_bedrooms": int(df["total_bedrooms"].isna().sum()),
    }
