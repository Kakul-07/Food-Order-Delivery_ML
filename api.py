import json
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

DELIVERY_MODEL_PATH = MODELS_DIR / "delivery_eta.joblib"
DELIVERY_METADATA_PATH = MODELS_DIR / "delivery_eta_metadata.json"

RESTAURANT_MODEL_PATH = (
    MODELS_DIR / "restaurant_recommendation_model.joblib"
)
PREVIOUS_ORDER_MODEL_PATH = (
    MODELS_DIR / "previous_order_model.joblib"
)

ORDERS_DATA_PATH = PROCESSED_DIR / "online_orders_clean.csv"
RESTAURANTS_DATA_PATH = PROCESSED_DIR / "restaurants_clean.csv"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Food Ordering and Delivery ML API",
    description=(
        "API for delivery-time prediction, restaurant recommendations, "
        "and customer order-history recommendations."
    ),
    version="1.0.0",
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

def load_joblib(path: Path):
    if not path.exists():
        print(f"[WARNING] Model artifact not found: {path}")
        return None

    try:
        artifact = joblib.load(path)
        print(f"[INFO] Loaded artifact: {path.name}")
        return artifact
    except Exception as exc:
        print(f"[ERROR] Could not load {path.name}: {exc}")
        return None


def load_json(path: Path):
    if not path.exists():
        print(f"[WARNING] Metadata file not found: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[ERROR] Could not load metadata: {exc}")
        return {}


def load_csv(path: Path):
    if not path.exists():
        print(f"[WARNING] CSV file not found: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[ERROR] Could not read {path.name}: {exc}")
        return pd.DataFrame()


delivery_model = load_joblib(DELIVERY_MODEL_PATH)
delivery_metadata = load_json(DELIVERY_METADATA_PATH)

restaurant_model_artifact = load_joblib(RESTAURANT_MODEL_PATH)
previous_order_model_artifact = load_joblib(PREVIOUS_ORDER_MODEL_PATH)

orders_df = load_csv(ORDERS_DATA_PATH)
restaurants_df = load_csv(RESTAURANTS_DATA_PATH)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = df.copy()
    result.columns = [normalize_column_name(c) for c in result.columns]
    return result


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """Find a column using possible names."""
    if df.empty:
        return None

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    return None


def to_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def get_restaurants() -> pd.DataFrame:
    """
    Load restaurant rows from the recommendation artifact when available.
    Fall back to restaurants_clean.csv.
    """
    if isinstance(restaurant_model_artifact, dict):
        records = restaurant_model_artifact.get("restaurants")

        if isinstance(records, pd.DataFrame):
            return normalize_dataframe(records)

        if isinstance(records, list):
            return normalize_dataframe(pd.DataFrame(records))

    if not restaurants_df.empty:
        return normalize_dataframe(restaurants_df)

    if not orders_df.empty:
        # Last-resort catalogue built from the order dataset.
        df = normalize_dataframe(orders_df)

        name_col = find_column(
            df, ["restaurant_name", "restaurant", "name"]
        )

        if name_col:
            return df.drop_duplicates(subset=[name_col]).reset_index(
                drop=True
            )

    return pd.DataFrame()


# ============================================================
# DELIVERY PREDICTION
# ============================================================

class DeliveryPredictionRequest(BaseModel):
    age: Optional[float] = Field(default=None, alias="Age")
    ratings: Optional[float] = Field(default=None, alias="Ratings")

    restaurant_latitude: Optional[float] = Field(
        default=None, alias="Restaurant_latitude"
    )
    restaurant_longitude: Optional[float] = Field(
        default=None, alias="Restaurant_longitude"
    )
    delivery_location_latitude: Optional[float] = Field(
        default=None, alias="Delivery_location_latitude"
    )
    delivery_location_longitude: Optional[float] = Field(
        default=None, alias="Delivery_location_longitude"
    )

    weather_conditions: Optional[str] = Field(
        default=None, alias="Weather_conditions"
    )
    road_traffic_density: Optional[str] = Field(
        default=None, alias="Road_traffic_density"
    )
    vehicle_condition: Optional[float] = Field(
        default=None, alias="Vehicle_condition"
    )
    type_of_order: Optional[str] = Field(
        default=None, alias="Type_of_order"
    )
    type_of_vehicle: Optional[str] = Field(
        default=None, alias="Type_of_vehicle"
    )
    multiple_deliveries: Optional[float] = Field(
        default=None, alias="multiple_deliveries"
    )
    festival: Optional[str] = Field(default=None, alias="Festival")
    city: Optional[str] = Field(default=None, alias="City")
    time_ordered: Optional[str] = Field(
        default=None, alias="Time_Orderd"
    )
    time_order_picked: Optional[str] = Field(
        default=None, alias="Time_Order_picked"
    )
    distance_km: Optional[float] = Field(
        default=None, alias="distance_km"
    )

    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }


def parse_order_time(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
        try:
            return pd.to_datetime(text, format=fmt)
        except (ValueError, TypeError):
            continue

    try:
        return pd.to_datetime(text)
    except (ValueError, TypeError):
        return None


def calculate_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """Calculate distance between two coordinates using Haversine."""
    values = [lat1, lon1, lat2, lon2]

    if any(value is None for value in values):
        return None

    try:
        lat1, lon1, lat2, lon2 = map(float, values)
    except (ValueError, TypeError):
        return None

    radius = 6371.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2) ** 2
    )

    return float(
        radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    )


def build_delivery_features(request: DeliveryPredictionRequest):
    """
    Construct model input from the request.

    If delivery metadata includes feature_columns, those columns determine
    the feature order. Categorical values are kept as strings.
    """
    payload = request.model_dump(
        by_alias=True,
        exclude_none=True,
    )

    payload = {
        normalize_column_name(key): value
        for key, value in payload.items()
    }

    distance = payload.get("distance_km")

    if distance is None:
        distance = calculate_distance_km(
            payload.get("restaurant_latitude"),
            payload.get("restaurant_longitude"),
            payload.get("delivery_location_latitude"),
            payload.get("delivery_location_longitude"),
        )

    if distance is not None:
        payload["distance_km"] = distance

    ordered_time = parse_order_time(payload.get("time_ordered"))
    picked_time = parse_order_time(payload.get("time_order_picked"))

    if ordered_time is not None:
        payload["order_hour"] = ordered_time.hour
        payload["order_minute"] = ordered_time.minute

    if ordered_time is not None and picked_time is not None:
        delta_minutes = (
            picked_time - ordered_time
        ).total_seconds() / 60

        if delta_minutes < 0:
            delta_minutes += 24 * 60

        payload["pickup_delay_minutes"] = delta_minutes

    feature_columns = delivery_metadata.get(
        "feature_columns",
        delivery_metadata.get("features", []),
    )

    if isinstance(feature_columns, list) and feature_columns:
        row = {}

        for column in feature_columns:
            normalized = normalize_column_name(column)
            row[column] = payload.get(
                normalized,
                payload.get(column),
            )

        return pd.DataFrame([row], columns=feature_columns)

    return pd.DataFrame([payload])


@app.post("/predict-delivery")
def predict_delivery(request: DeliveryPredictionRequest):
    if delivery_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Delivery model is not loaded. Check "
                "models/delivery_eta.joblib."
            ),
        )

    try:
        features = build_delivery_features(request)

        # Pipeline artifacts generally expose predict().
        if not hasattr(delivery_model, "predict"):
            raise HTTPException(
                status_code=500,
                detail="The loaded delivery artifact has no predict() method.",
            )

        prediction = delivery_model.predict(features)
        predicted_minutes = float(np.asarray(prediction).reshape(-1)[0])

        return {
            "predicted_delivery_time_minutes": round(
                predicted_minutes, 2
            ),
            "unit": "minutes",
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[ERROR] Delivery prediction failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Delivery prediction failed: {str(exc)}",
        )


# ============================================================
# RESTAURANT RECOMMENDATIONS
# ============================================================

@app.get("/restaurants/recommend")
def recommend_restaurants(
    cuisine: Optional[str] = Query(default=None),
    location: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
):
    restaurants = get_restaurants()

    if restaurants.empty:
        raise HTTPException(
            status_code=503,
            detail="Restaurant catalogue is unavailable.",
        )

    name_col = find_column(
        restaurants,
        ["restaurant_name", "restaurant", "name"],
    )
    cuisine_col = find_column(
        restaurants,
        ["cuisine", "cuisine_type", "cuisines"],
    )
    location_col = find_column(
        restaurants,
        ["location", "city", "address", "locality"],
    )
    rating_col = find_column(
        restaurants,
        ["rating", "ratings", "aggregate_rating"],
    )

    if name_col is None:
        raise HTTPException(
            status_code=500,
            detail="Restaurant name column is missing.",
        )

    df = restaurants.copy()
    df["_original_index"] = df.index

    if cuisine and cuisine_col:
        df = df[
            df[cuisine_col]
            .astype(str)
            .str.contains(cuisine, case=False, na=False)
        ]

    if location and location_col:
        df = df[
            df[location_col]
            .astype(str)
            .str.contains(location, case=False, na=False)
        ]

    if df.empty:
        return {
            "count": 0,
            "recommendations": [],
            "message": "No restaurants match the supplied filters.",
        }

    artifact = restaurant_model_artifact or {}

    vectorizer = (
        artifact.get("vectorizer")
        if isinstance(artifact, dict)
        else None
    )
    restaurant_matrix = (
        artifact.get("restaurant_matrix")
        if isinstance(artifact, dict)
        else None
    )
    artifact_rows = (
        artifact.get("restaurants")
        if isinstance(artifact, dict)
        else None
    )

    scores = {}

    if (
        vectorizer is not None
        and restaurant_matrix is not None
        and artifact_rows is not None
        and cuisine
    ):
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vector = vectorizer.transform([cuisine])
            similarities = cosine_similarity(
                query_vector, restaurant_matrix
            ).reshape(-1)

            for original_index in df["_original_index"]:
                if 0 <= int(original_index) < len(similarities):
                    scores[int(original_index)] = float(
                        similarities[int(original_index)]
                    )
        except Exception as exc:
            print(
                "[WARNING] Content-based recommendation failed: "
                f"{exc}"
            )

    def score_restaurant(row):
        index = int(row["_original_index"])
        content_score = scores.get(index, 0.0)

        rating = (
            to_float(row.get(rating_col), 0.0)
            if rating_col
            else 0.0
        )

        # Normalize a rating commonly expressed on a 5-point scale.
        rating_score = min(max(rating / 5.0, 0.0), 1.0)

        return 0.65 * content_score + 0.35 * rating_score

    df["_recommendation_score"] = df.apply(
        score_restaurant, axis=1
    )

    df = df.sort_values(
        "_recommendation_score",
        ascending=False,
    ).head(limit)

    results = []

    for _, row in df.iterrows():
        item = {
            "restaurant_name": str(row[name_col]),
            "recommendation_score": round(
                float(row["_recommendation_score"]), 4
            ),
        }

        if cuisine_col:
            item["cuisine"] = row.get(cuisine_col)

        if location_col:
            item["location"] = row.get(location_col)

        if rating_col:
            item["rating"] = row.get(rating_col)

        results.append(item)

    return {
        "count": len(results),
        "recommendations": results,
    }


# ============================================================
# PREVIOUS-ORDER RECOMMENDATIONS
# ============================================================

@app.get("/restaurants/previous-orders")
def recommend_from_previous_orders(
    customer_id: str = Query(...),
    limit: int = Query(default=10, ge=1, le=50),
):
    if previous_order_model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Previous-order model artifact is not loaded.",
        )

    artifact = previous_order_model_artifact

    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=500,
            detail="Previous-order artifact has an unexpected format.",
        )

    # These fields come from the previous-order training artifact.
    customer_column = artifact.get("customer_column")
    restaurant_column = artifact.get("restaurant_column")
    cuisine_column = artifact.get("cuisine_column")

    # FIX: Restaurant catalogue column names must be obtained from the
    # restaurant artifact or detected in the catalogue, not from the
    # previous-order artifact's nonexistent "columns" dictionary.
    restaurants = get_restaurants()

    if restaurants.empty:
        raise HTTPException(
            status_code=503,
            detail="Restaurant catalogue is unavailable.",
        )

    restaurant_columns = (
        restaurant_model_artifact.get("columns", {})
        if isinstance(restaurant_model_artifact, dict)
        else {}
    )

    name_col = restaurant_columns.get("name")
    cuisine_col = restaurant_columns.get("cuisine")
    rating_col = restaurant_columns.get("rating")
    location_col = restaurant_columns.get("location")

    # Fall back to detecting the column names from the actual dataframe.
    if not name_col or name_col not in restaurants.columns:
        name_col = find_column(
            restaurants,
            ["restaurant_name", "restaurant", "name"],
        )

    if not cuisine_col or cuisine_col not in restaurants.columns:
        cuisine_col = find_column(
            restaurants,
            ["cuisine", "cuisine_type", "cuisines"],
        )

    if not rating_col or rating_col not in restaurants.columns:
        rating_col = find_column(
            restaurants,
            ["rating", "ratings", "aggregate_rating"],
        )

    if not location_col or location_col not in restaurants.columns:
        location_col = find_column(
            restaurants,
            ["location", "city", "address", "locality"],
        )

    if name_col is None:
        raise HTTPException(
            status_code=500,
            detail="Restaurant name column is missing.",
        )

    # Match customer IDs as strings to avoid integer/string mismatches.
    customer_counts = artifact.get(
        "customer_restaurant_counts", {}
    )
    cuisine_preferences = artifact.get(
        "customer_cuisine_preferences", {}
    )

    customer_key = str(customer_id).strip()

    customer_restaurant_history = None
    for stored_customer_id, history in customer_counts.items():
        if str(stored_customer_id).strip() == customer_key:
            customer_restaurant_history = history
            break

    customer_cuisine_history = None
    for stored_customer_id, history in cuisine_preferences.items():
        if str(stored_customer_id).strip() == customer_key:
            customer_cuisine_history = history
            break

    if customer_restaurant_history is None:
        raise HTTPException(
            status_code=404,
            detail=f"No order history found for customer {customer_id}.",
        )

    # Support histories represented as either dictionaries or lists.
    if isinstance(customer_restaurant_history, dict):
        restaurant_counts = customer_restaurant_history
    elif isinstance(customer_restaurant_history, list):
        restaurant_counts = {}

        for item in customer_restaurant_history:
            if isinstance(item, dict):
                restaurant_name = (
                    item.get("restaurant_name")
                    or item.get("restaurant")
                    or item.get("name")
                )
                count = item.get("count", item.get("orders", 1))

                if restaurant_name:
                    restaurant_counts[str(restaurant_name)] = (
                        to_float(count, 1.0)
                    )
    else:
        restaurant_counts = {}

    if isinstance(customer_cuisine_history, dict):
        cuisine_counts = customer_cuisine_history
    elif isinstance(customer_cuisine_history, list):
        cuisine_counts = {
            str(cuisine): 1
            for cuisine in customer_cuisine_history
        }
    else:
        cuisine_counts = {}

    df = restaurants.copy()

    # Calculate popularity from the saved restaurant catalogue where possible.
    popularity = artifact.get("restaurant_popularity", {})
    popularity_by_name = {
        str(key).strip().lower(): to_float(value, 0.0)
        for key, value in popularity.items()
    }

    previously_ordered = {
        str(name).strip().lower()
        for name in restaurant_counts.keys()
    }

    def history_score(row):
        name = str(row.get(name_col, "")).strip()
        normalized_name = name.lower()

        restaurant_count = to_float(
            restaurant_counts.get(name, 0),
            0.0,
        )

        if restaurant_count == 0:
            for stored_name, stored_count in restaurant_counts.items():
                if str(stored_name).strip().lower() == normalized_name:
                    restaurant_count = to_float(stored_count, 0.0)
                    break

        cuisine = (
            str(row.get(cuisine_col, "")).strip()
            if cuisine_col
            else ""
        )

        cuisine_count = to_float(
            cuisine_counts.get(cuisine, 0),
            0.0,
        )

        if cuisine_count == 0:
            for stored_cuisine, stored_count in cuisine_counts.items():
                if str(stored_cuisine).strip().lower() == cuisine.lower():
                    cuisine_count = to_float(stored_count, 0.0)
                    break

        rating = (
            to_float(row.get(rating_col), 0.0)
            if rating_col
            else 0.0
        )

        popularity_score = popularity_by_name.get(
            normalized_name, 0.0
        )

        return (
            0.50 * restaurant_count
            + 0.25 * cuisine_count
            + 0.15 * min(max(rating / 5.0, 0.0), 1.0)
            + 0.10 * popularity_score
        )

    df["_history_score"] = df.apply(history_score, axis=1)

    # Prefer restaurants not already ordered when alternatives exist.
    new_restaurants = df[
        ~df[name_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(previously_ordered)
    ]

    if not new_restaurants.empty:
        df = new_restaurants

    df = df.sort_values(
        "_history_score",
        ascending=False,
    ).head(limit)

    recommendations = []

    for _, row in df.iterrows():
        item = {
            "restaurant_name": str(row[name_col]),
            "recommendation_score": round(
                float(row["_history_score"]), 4
            ),
        }

        if cuisine_col:
            item["cuisine"] = row.get(cuisine_col)

        if rating_col:
            item["rating"] = row.get(rating_col)

        if location_col:
            item["location"] = row.get(location_col)

        recommendations.append(item)

    return {
        "customer_id": customer_id,
        "count": len(recommendations),
        "recommendations": recommendations,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    restaurants = get_restaurants()

    return {
        "status": "ok",
        "delivery_model_loaded": delivery_model is not None,
        "restaurant_model_loaded": (
            restaurant_model_artifact is not None
        ),
        "previous_order_model_loaded": (
            previous_order_model_artifact is not None
        ),
        "restaurant_data_available": not restaurants.empty,
        "order_history_available": not orders_df.empty,
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Food Ordering and Delivery ML API is running.",
        "docs": "/docs",
        "health": "/health",
    }