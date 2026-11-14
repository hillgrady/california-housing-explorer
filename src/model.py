"""House price model: training, validation of inputs, prediction and explanation.

The model is a histogram gradient boosting regressor (scikit-learn) with two
quantile models (10% and 90%) that give an 80% prediction range. It is trained
on 80% of the districts and evaluated on the other 20%.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import CATEGORIES, VALUE_CAP

NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
           "population", "households", "median_income"]
CAT = "ocean_proximity"
FEATURES = NUMERIC + [CAT]
TARGET = "median_house_value"
EARTH_KM = 6371.0
MAX_DISTANCE_KM = 25.0  # inputs farther than this from any 1990 district are rejected

# Simplified California border (longitude, latitude). The coast side is loose on
# purpose; points in the ocean are caught by the distance check instead.
CA_BORDER = [(-125.0, 42.0), (-120.0, 42.0), (-120.0, 39.0), (-114.63, 35.0), (-114.13, 34.27),
             (-114.72, 32.72), (-117.13, 32.53), (-125.0, 32.53)]


def in_california(lat: float, lon: float) -> bool:
    """Ray casting point in polygon test against CA_BORDER."""
    inside = False
    pts = CA_BORDER
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def _pipeline(**kw) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(categories=[CATEGORIES], handle_unknown="use_encoded_value",
                               unknown_value=-1), [CAT]),
        ("num", "passthrough", NUMERIC),
    ])
    hgb = HistGradientBoostingRegressor(categorical_features=[0], max_iter=400,
                                        learning_rate=0.08, random_state=42, **kw)
    return Pipeline([("pre", pre), ("hgb", hgb)])


@dataclass
class PriceModel:
    point: Pipeline
    low: Pipeline
    high: Pipeline
    metrics: dict
    ranges: dict
    tree: BallTree
    reference: pd.DataFrame = field(repr=False)

    # ------------------------------------------------------------------ checks
    def nearest_km(self, lat: float, lon: float) -> float:
        d, _ = self.tree.query(np.radians([[lat, lon]]), k=1)
        return float(d[0, 0] * EARTH_KM)

    def validate(self, row: dict) -> tuple[list[str], list[str]]:
        """Return (errors, warnings). Errors block the prediction."""
        errors, warnings = [], []
        dist = self.nearest_km(row["latitude"], row["longitude"])
        if not in_california(row["latitude"], row["longitude"]):
            errors.append("This location is outside California. The model only covers California "
                          "districts from the 1990 Census, so it cannot price this location.")
        elif dist > MAX_DISTANCE_KM:
            errors.append(
                f"This location is {dist:,.0f} km from the nearest California district in the data. "
                "The model only covers California in 1990, so it cannot price this location.")
        if row[CAT] not in CATEGORIES:
            errors.append(f"Ocean proximity must be one of {', '.join(CATEGORIES)}.")
        if row["households"] <= 0 or row["total_rooms"] <= 0:
            errors.append("Households and rooms must be positive.")
        for col in ["housing_median_age", "median_income", "total_rooms", "population", "households"]:
            lo, hi = self.ranges[col]
            if not lo <= row[col] <= hi:
                warnings.append(f"{col.replace('_', ' ')} = {row[col]:,.2f} is outside the range seen in "
                                f"training ({lo:,.2f} to {hi:,.2f}); the prediction is less reliable.")
        return errors, warnings

    # -------------------------------------------------------------- predicting
    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES]
        p = self.point.predict(X)
        lo = np.minimum(self.low.predict(X), p)
        hi = np.maximum(self.high.predict(X), p)
        cap = lambda a: np.clip(a, 14_999, VALUE_CAP)  # noqa: E731
        return pd.DataFrame({"prediction": cap(p), "low": cap(lo), "high": cap(hi)}, index=X.index)

    def predict(self, row: dict) -> dict:
        out = self.predict_frame(pd.DataFrame([row])).iloc[0]
        return {k: float(v) for k, v in out.items()}

    def sensitivity(self, row: dict) -> pd.DataFrame:
        """How the prediction moves when one input changes (all else fixed)."""
        base = self.predict(row)["prediction"]
        scenarios = [
            ("Median income +$10,000", "median_income", +1.0),
            ("Median income -$10,000", "median_income", -1.0),
            ("Houses 10 years older", "housing_median_age", +10),
            ("Houses 10 years newer", "housing_median_age", -10),
        ]
        rows = []
        for label, col, delta in scenarios:
            r = dict(row)
            lo, hi = self.ranges[col]
            r[col] = float(np.clip(r[col] + delta, lo, hi))
            rows.append({"Change": label, "New prediction": self.predict(r)["prediction"]})
        for cat in CATEGORIES:
            if cat != row[CAT]:
                r = dict(row, **{CAT: cat})
                rows.append({"Change": f"Ocean proximity = {cat}", "New prediction": self.predict(r)["prediction"]})
        out = pd.DataFrame(rows)
        out["Difference"] = out["New prediction"] - base
        return out

    def similar_districts(self, row: dict, k: int = 5) -> pd.DataFrame:
        """Actual 1990 districts nearest to the input location."""
        _, idx = self.tree.query(np.radians([[row["latitude"], row["longitude"]]]), k=k)
        cols = ["latitude", "longitude", "ocean_proximity", "median_income",
                "housing_median_age", "median_house_value"]
        return self.reference.iloc[idx[0]][cols].reset_index(drop=True)


def train(df: pd.DataFrame) -> PriceModel:
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    point = _pipeline().fit(Xtr, ytr)
    low = _pipeline(loss="quantile", quantile=0.10).fit(Xtr, ytr)
    high = _pipeline(loss="quantile", quantile=0.90).fit(Xtr, ytr)
    pred = point.predict(Xte)
    lo, hi = low.predict(Xte), high.predict(Xte)
    metrics = {
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
        "coverage_80": float(np.mean((yte >= np.minimum(lo, pred)) & (yte <= np.maximum(hi, pred)))),
    }
    ranges = {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC}
    ref = df.reset_index(drop=True)
    tree = BallTree(np.radians(ref[["latitude", "longitude"]].to_numpy()), metric="haversine")
    return PriceModel(point, low, high, metrics, ranges, tree, ref)


def build_row(latitude, longitude, ocean_proximity, median_income, housing_median_age,
              households, rooms_per_household, bedrooms_per_room, people_per_household) -> dict:
    """Turn the per household inputs shown in the app into the district totals the model uses."""
    total_rooms = rooms_per_household * households
    return {
        "longitude": float(longitude), "latitude": float(latitude),
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(bedrooms_per_room * total_rooms),
        "population": float(people_per_household * households),
        "households": float(households),
        "median_income": float(median_income),
        "ocean_proximity": ocean_proximity,
    }
