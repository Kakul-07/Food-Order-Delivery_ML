"""FastAPI endpoints for the Food Ordering & Delivery ML project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "delivery_eta.joblib"
METADATA_PATH = ROOT / "models" / "delivery_eta_metadata.json"

RESTAURANTS_PATH = (
    ROOT / "data" / "processed" / "restaurants_clean.csv"
)

ORDER_HISTORY_PATH = (
    ROOT / "data" / "raw" / "online_food_orders.csv"
)

CLEAN_ORDER_HISTORY_PATH = (
    ROOT / "data" / "processed" / "online_orders_clean.csv"
)


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Food Ordering & Delivery ML API",
    description=(
        "API for delivery-time prediction, restaurant ranking, "
        "and previous-order-based recommendations."
    ),
    version="1.0.0",
)

# Frontend calls are browser-originated. Restrict this list to your deployed
# frontend domain(s) with FRONTEND_ORIGINS in production, comma-separated.
# The wildcard fallback is convenient for initial testing; it does not enable
# credentialed cookies.
import os

_frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

model = None
metadata = {}

if MODEL_PATH.exists() and METADATA_PATH.exists():
    model = joblib.load(MODEL_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )


# ============================================================
# 4. DELIVERY PREDICTION REQUEST
# ============================================================

class DeliveryPredictionRequest(BaseModel):
    """Validated request body for delivery-time prediction.

    Fields are optional for compatibility with the existing API. The model
    preprocessing layer handles missing values using the same approach as
    the original service. Frontend-provided route distance is preferred.
    """

    Delivery_person_Age: Optional[float] = Field(default=None, ge=18, le=80)
    Delivery_person_Ratings: Optional[float] = Field(default=None, ge=0, le=5)

    Restaurant_latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    Restaurant_longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    Delivery_location_latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    Delivery_location_longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    Vehicle_condition: Optional[float] = Field(default=None, ge=0, le=5)

    # Route distance from the frontend Maps API, in kilometres.
    distance_km: Optional[float] = Field(default=None, gt=0, le=1000)

    # Explicit time features; when omitted, derive them from time/date fields.
    order_hour: Optional[int] = Field(default=None, ge=0, le=23)
    order_minute: Optional[int] = Field(default=None, ge=0, le=59)
    order_weekday: Optional[int] = Field(default=None, ge=0, le=6)
    order_month: Optional[int] = Field(default=None, ge=1, le=12)

    Time_Orderd: Optional[str] = None
    Order_Date: Optional[str] = None

    # Values are supplied by the frontend Weather API / existing traffic source.
    Weatherconditions: Optional[str] = None
    Road_traffic_density: Optional[str] = None

    Type_of_order: Optional[str] = None
    Type_of_vehicle: Optional[str] = None

    Festival: Optional[str] = None
    City: Optional[str] = None


# ============================================================
# 5. HELPER FUNCTIONS FOR DELIVERY PREDICTION
# ============================================================

def parse_order_time(value):
    """Convert an order time into hour and minute."""

    if value is None or pd.isna(value):
        return np.nan, np.nan

    match = re.match(
        r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)?\s*$",
        str(value),
        re.I,
    )

    if not match:
        return np.nan, np.nan

    hour = int(match.group(1))
    minute = int(match.group(2))

    suffix = (match.group(3) or "").upper()

    if suffix == "PM" and hour < 12:
        hour += 12

    if suffix == "AM" and hour == 12:
        hour = 0

    if hour > 23 or minute > 59:
        return np.nan, np.nan

    return hour, minute


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare input features for the trained delivery model."""

    x = df.copy()

    numeric_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Vehicle_condition",
    ]

    for col in numeric_columns:
        if col not in x.columns:
            x[col] = np.nan

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    # Prefer route distance supplied by the frontend Maps API.
    # Fall back to straight-line Haversine distance if it is not supplied.
    if "distance_km" not in x.columns:
        x["distance_km"] = np.nan

    x["distance_km"] = pd.to_numeric(
        x["distance_km"],
        errors="coerce",
    )

    lat1 = np.radians(x["Restaurant_latitude"])
    lon1 = np.radians(x["Restaurant_longitude"])
    lat2 = np.radians(x["Delivery_location_latitude"])
    lon2 = np.radians(x["Delivery_location_longitude"])

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )

    straight_line_distance = (
        6371
        * 2
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )

    x["distance_km"] = x["distance_km"].where(
        x["distance_km"].notna(),
        straight_line_distance,
    )

    # Derive time features, then prefer explicit frontend values when present.
    if "Time_Orderd" not in x.columns:
        x["Time_Orderd"] = np.nan

    if "Order_Date" not in x.columns:
        x["Order_Date"] = np.nan

    time_parts = x["Time_Orderd"].apply(parse_order_time)
    derived_hour = time_parts.map(lambda value: value[0])
    derived_minute = time_parts.map(lambda value: value[1])

    x["order_hour"] = pd.to_numeric(
        x.get("order_hour", pd.Series(np.nan, index=x.index)),
        errors="coerce",
    ).where(
        pd.to_numeric(
            x.get("order_hour", pd.Series(np.nan, index=x.index)),
            errors="coerce",
        ).notna(),
        derived_hour,
    )

    x["order_minute"] = pd.to_numeric(
        x.get("order_minute", pd.Series(np.nan, index=x.index)),
        errors="coerce",
    ).where(
        pd.to_numeric(
            x.get("order_minute", pd.Series(np.nan, index=x.index)),
            errors="coerce",
        ).notna(),
        derived_minute,
    )

    dates = pd.to_datetime(
        x["Order_Date"],
        errors="coerce",
        dayfirst=True,
    )
    derived_weekday = dates.dt.dayofweek
    derived_month = dates.dt.month

    for col, derived in [
        ("order_weekday", derived_weekday),
        ("order_month", derived_month),
    ]:
        supplied = pd.to_numeric(
            x.get(col, pd.Series(np.nan, index=x.index)),
            errors="coerce",
        )
        x[col] = supplied.where(supplied.notna(), derived)

    # Clean categorical features.

    categorical_columns = [
        "Weatherconditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City",
    ]

    for col in categorical_columns:
        if col not in x.columns:
            x[col] = np.nan

        x[col] = (
            x[col]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "": pd.NA,
                    "NaN": pd.NA,
                    "nan": pd.NA,
                    "conditions NaN": pd.NA,
                }
            )
            .astype(object)
        )

        x[col] = x[col].where(
            pd.notna(x[col]),
            np.nan,
        )

    # Use exactly the features expected by the trained model.

    feature_columns = metadata.get("feature_columns")

    if not feature_columns:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model metadata is missing. "
                "Run notebook 04 to train and export the model."
            ),
        )

    for col in feature_columns:
        if col not in x.columns:
            x[col] = np.nan

    return x[feature_columns]


# ============================================================
# 6. LOAD RESTAURANT DATA
# ============================================================

def get_restaurants() -> pd.DataFrame:
    """Load the cleaned restaurant listing dataset."""

    if not RESTAURANTS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Clean restaurant data is missing. "
                "Run notebook 02_cleaning_and_preprocessing.ipynb."
            ),
        )

    return pd.read_csv(
        RESTAURANTS_PATH,
        low_memory=False,
    )


# ============================================================
# 7. LOAD CUSTOMER ORDER HISTORY
# ============================================================

def get_order_history() -> pd.DataFrame:
    """Load historical orders for cuisine recommendations."""

    if CLEAN_ORDER_HISTORY_PATH.exists():
        path = CLEAN_ORDER_HISTORY_PATH
    else:
        path = ORDER_HISTORY_PATH

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Order history is missing. Add "
                "data/raw/online_food_orders.csv or generate "
                "data/processed/online_orders_clean.csv."
            ),
        )

    orders = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "customer_id",
        "cuisine_type",
    }

    if not required_columns.issubset(orders.columns):
        raise HTTPException(
            status_code=500,
            detail=(
                "Order history must contain customer_id "
                "and cuisine_type columns."
            ),
        )

    return orders


def normalize_id(value) -> str:
    """Normalize customer IDs read from CSV files."""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


# ============================================================
# 8. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Food Ordering & Delivery ML API",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# 9. HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "delivery_model_loaded": model is not None,
        "restaurant_data_available": RESTAURANTS_PATH.exists(),
    }


# ============================================================
# 10. DELIVERY TIME PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-delivery")
def predict_delivery(
    request: DeliveryPredictionRequest,
):
    """Predict delivery duration in minutes."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Delivery model is not trained or exported. "
                "Run notebooks 01–04 to generate the model files."
            ),
        )

    payload = (
        request.model_dump()
        if hasattr(request, "model_dump")
        else request.dict()
    )

    features = build_features(
        pd.DataFrame([payload])
    )

    prediction = float(
        np.clip(
            model.predict(features)[0],
            1,
            180,
        )
    )

    return {
        "predicted_delivery_minutes": round(
            prediction,
            1,
        ),
        "unit": "minutes",
    }


# ============================================================
# 11. RESTAURANT RECOMMENDATION ENDPOINT
# ============================================================

@app.get("/restaurants/recommend")
def recommend_restaurants(
    cuisine: Optional[str] = None,
    location: Optional[str] = None,
    max_price: Optional[float] = Query(
        default=None,
        ge=0,
    ),
    min_rating: Optional[float] = Query(
        default=None,
        ge=0,
        le=5,
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
):
    """Filter and rank restaurants by rating and popularity."""

    restaurants = get_restaurants().copy()

    # Ensure expected columns exist.

    for col in [
        "restaurant_name",
        "cuisine",
        "area",
        "location",
    ]:
        if col not in restaurants.columns:
            restaurants[col] = ""

        restaurants[col] = (
            restaurants[col]
            .fillna("")
            .astype(str)
        )

    for col in [
        "rating",
        "rating_count",
        "average_price",
    ]:
        if col not in restaurants.columns:
            restaurants[col] = np.nan

        restaurants[col] = pd.to_numeric(
            restaurants[col],
            errors="coerce",
        )

    restaurants["rating_count"] = (
        restaurants["rating_count"].fillna(0)
    )

    ratings = restaurants["rating"].dropna()

    global_mean = (
        float(ratings.mean())
        if not ratings.empty
        else 0.0
    )

    # Bayesian-style ranking score.

    m = (
        float(
            restaurants["rating_count"].quantile(0.75)
        )
        if not restaurants.empty
        else 0.0
    )

    if m > 0:
        denominator = restaurants["rating_count"] + m

        restaurants["ranking_score"] = (
            restaurants["rating_count"]
            / denominator
            * restaurants["rating"].fillna(global_mean)
            + m
            / denominator
            * global_mean
        )
    else:
        restaurants["ranking_score"] = (
            restaurants["rating"].fillna(global_mean)
        )

    # Apply filters.

    if cuisine:
        restaurants = restaurants[
            restaurants["cuisine"].str.contains(
                cuisine,
                case=False,
                na=False,
            )
        ]

    if location:
        location_mask = (
            restaurants["location"].str.contains(
                location,
                case=False,
                na=False,
            )
            | restaurants["area"].str.contains(
                location,
                case=False,
                na=False,
            )
        )

        restaurants = restaurants[location_mask]

    if max_price is not None:
        restaurants = restaurants[
            restaurants["average_price"].le(max_price)
        ]

    if min_rating is not None:
        restaurants = restaurants[
            restaurants["rating"].ge(min_rating)
        ]

    # Sort and limit results.

    restaurants = restaurants.sort_values(
        ["ranking_score", "rating_count"],
        ascending=False,
    ).head(limit)

    output_columns = [
        col
        for col in [
            "restaurant_name",
            "cuisine",
            "rating",
            "rating_count",
            "average_price",
            "area",
            "location",
            "ranking_score",
        ]
        if col in restaurants.columns
    ]

    records = (
        restaurants[output_columns]
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    return {
        "count": len(records),
        "restaurants": records,
        "ranking": "static rating/popularity score",
    }


# ============================================================
# 12. PREVIOUS-ORDER-BASED RESTAURANT RECOMMENDATION
# ============================================================

@app.get("/restaurants/previous-orders")
def recommend_from_previous_orders(
    customer_id: str = Query(
        ...,
        min_length=1,
        description="Customer ID present in the order-history dataset",
    ),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Personalize restaurant recommendations using a customer's order history.

    The endpoint learns cuisine preferences from the customer's historical
    orders, ranks cuisines by frequency, and combines recommendations from
    the strongest preferences. It gracefully handles datasets that do not
    contain order dates or restaurant names.
    """
    orders = get_order_history().copy()
    orders["customer_id_normalized"] = orders["customer_id"].map(normalize_id)
    customer_orders = orders[
        orders["customer_id_normalized"].eq(normalize_id(customer_id))
    ].copy()

    if customer_orders.empty:
        raise HTTPException(
            status_code=404,
            detail=(f"No previous orders found for customer_id '{customer_id}' "
                    "in the available dataset."),
        )

    customer_orders["cuisine_type"] = (
        customer_orders["cuisine_type"].fillna("").astype(str).str.strip()
    )
    customer_orders = customer_orders[customer_orders["cuisine_type"].ne("")]
    if customer_orders.empty:
        raise HTTPException(
            status_code=404,
            detail="Previous orders were found, but none had a recorded cuisine type.",
        )

    # If an order date/time column exists, sort newest-first and apply a mild
    # recency weighting; otherwise use frequency only (works with simpler CSVs).
    date_col = next((c for c in ["order_date", "Order_Date", "order_timestamp",
                                  "order_time", "created_at", "timestamp"]
                     if c in customer_orders.columns), None)
    if date_col:
        dates = pd.to_datetime(customer_orders[date_col], errors="coerce")
        if dates.notna().any():
            customer_orders["_order_date"] = dates
            customer_orders = customer_orders.sort_values("_order_date")
            # Newer rows receive larger weights. A simple rank weight avoids
            # assumptions about the date format or elapsed-time scale.
            customer_orders["_recency_weight"] = np.arange(
                1, len(customer_orders) + 1, dtype=float
            )
        else:
            customer_orders["_recency_weight"] = 1.0
    else:
        customer_orders["_recency_weight"] = 1.0

    cuisine_stats = (
        customer_orders.groupby("cuisine_type", as_index=False)
        .agg(order_count=("cuisine_type", "size"),
             preference_score=("_recency_weight", "sum"))
        .sort_values(["preference_score", "order_count"], ascending=False)
    )
    top_cuisines = cuisine_stats.head(min(3, len(cuisine_stats)))

    # Allocate a recommendation quota across the user's top cuisine preferences.
    # Dedupe by restaurant name (or the full record if no name is available).
    collected = []
    seen = set()
    for _, cuisine_row in top_cuisines.iterrows():
        cuisine = str(cuisine_row["cuisine_type"])
        quota = max(1, limit // len(top_cuisines))
        result = recommend_restaurants(
            cuisine=cuisine, location=None, max_price=None,
            min_rating=None, limit=min(100, quota + 3),
        )
        for restaurant in result.get("restaurants", []):
            name = str(restaurant.get("restaurant_name", "")).strip().casefold()
            key = name if name else json.dumps(restaurant, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            restaurant = dict(restaurant)
            restaurant["matched_preferred_cuisine"] = cuisine
            restaurant["customer_cuisine_order_count"] = int(cuisine_row["order_count"])
            collected.append(restaurant)
            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break

    return {
        "customer_id": normalize_id(customer_id),
        "total_historical_orders_with_cuisine": int(len(customer_orders)),
        "preferred_cuisines": [
            {"cuisine": str(row["cuisine_type"]),
             "order_count": int(row["order_count"]),
             "preference_score": round(float(row["preference_score"]), 3)}
            for _, row in top_cuisines.iterrows()
        ],
        "count": len(collected),
        "restaurants": collected,
        "personalization": "ranked using customer cuisine frequency and, when available, order recency",
        "message": (
            "Recommendations are based on this customer's historical cuisine preferences."
            if collected else
            "Customer history was found, but no listed restaurants matched the customer's preferred cuisines."
        ),
    }

