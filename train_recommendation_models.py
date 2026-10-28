from pathlib import Path
import json
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

RESTAURANTS_PATH = DATA_DIR / "restaurants_clean.csv"
ORDERS_PATH = DATA_DIR / "online_orders_clean.csv"


def load_csv(path):
    """Load a CSV and normalize column names."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    df.columns = [
        re.sub(r"\s+", "_", str(col).strip().lower())
        for col in df.columns
    ]

    print(f"\nLoaded: {path.name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def find_column(df, candidates, required=True):
    """Find a column using possible column names."""
    for name in candidates:
        normalized = re.sub(
            r"\s+", "_", name.strip().lower()
        )

        if normalized in df.columns:
            return normalized

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return None


def train_restaurant_model():
    print("\nTraining restaurant recommendation model...")

    df = load_csv(RESTAURANTS_PATH)

    name_col = find_column(
        df,
        ["restaurant_name", "restaurant", "name"]
    )

    cuisine_col = find_column(
        df,
        ["cuisine", "cuisine_type", "cuisines"],
        required=False
    )

    location_col = find_column(
        df,
        ["location", "city", "area", "locality"],
        required=False
    )

    rating_col = find_column(
        df,
        ["rating", "aggregate_rating", "avg_rating"],
        required=False
    )

    price_col = find_column(
        df,
        ["average_price", "avg_price", "price", "cost_for_two"],
        required=False
    )

    # Remove invalid restaurant names.
    df = df.dropna(subset=[name_col]).copy()

    df[name_col] = df[name_col].astype(str).str.strip()

    df = df[
        df[name_col].ne("")
        & df[name_col].ne("nan")
    ]

    df = df.drop_duplicates(
        subset=[name_col]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid restaurants found.")

    # Prepare textual features.
    feature_columns = [
        col
        for col in [cuisine_col, location_col]
        if col is not None
    ]

    if feature_columns:
        for col in feature_columns:
            df[col] = df[col].fillna("").astype(str)

        text = df[feature_columns].agg(" ".join, axis=1)

    else:
        text = df[name_col].astype(str)

    text = text.replace("", "unknown").fillna("unknown")

    # Fit TF-IDF.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    restaurant_matrix = vectorizer.fit_transform(text)

    # Save model artifact.
    artifact = {
        "model_type": "tfidf_content_recommendation",
        "vectorizer": vectorizer,
        "restaurant_matrix": restaurant_matrix,
        "restaurants": df.to_dict(orient="records"),
        "columns": {
            "name": name_col,
            "cuisine": cuisine_col,
            "location": location_col,
            "rating": rating_col,
            "price": price_col
        },
        "training_rows": len(df)
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        MODELS_DIR / "restaurant_recommendation_model.joblib"
    )

    joblib.dump(artifact, output_path)

    print(f"Restaurant model saved: {output_path}")
    print(f"Restaurants used: {len(df)}")

    return {
        "model": str(output_path),
        "restaurants": len(df)
    }


def train_previous_order_model():
    print("\nTraining previous-order recommendation model...")

    df = load_csv(ORDERS_PATH)

    customer_col = find_column(
        df,
        ["customer_id", "user_id", "customer"]
    )

    restaurant_col = find_column(
        df,
        ["restaurant_name", "restaurant", "restaurant_id"]
    )

    cuisine_col = find_column(
        df,
        ["cuisine_type", "cuisine", "cuisines"],
        required=False
    )

    # Remove records without usable IDs or restaurant names.
    df = df.dropna(
        subset=[customer_col, restaurant_col]
    ).copy()

    df[customer_col] = df[customer_col].astype(str).str.strip()
    df[restaurant_col] = (
        df[restaurant_col].astype(str).str.strip()
    )

    df = df[
        df[customer_col].ne("")
        & df[restaurant_col].ne("")
        & df[customer_col].ne("nan")
        & df[restaurant_col].ne("nan")
    ]

    if df.empty:
        raise ValueError("No valid order-history records found.")

    # Count how often each customer ordered from each restaurant.
    interactions = (
        df.groupby([customer_col, restaurant_col])
        .size()
        .reset_index(name="order_count")
    )

    customer_restaurant_counts = {}

    for customer, group in interactions.groupby(customer_col):
        customer_restaurant_counts[str(customer)] = {
            str(row[restaurant_col]): int(row["order_count"])
            for _, row in group.iterrows()
        }

    # Overall restaurant popularity.
    popularity = (
        df[restaurant_col]
        .value_counts()
        .astype(int)
        .to_dict()
    )

    # Learn customer cuisine preferences.
    customer_cuisine_preferences = {}
    restaurant_cuisine = {}

    if cuisine_col:
        df[cuisine_col] = df[cuisine_col].fillna("").astype(str)

        for customer, group in df.groupby(customer_col):
            preferences = (
                group[cuisine_col]
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .to_dict()
            )

            customer_cuisine_preferences[str(customer)] = (
                preferences
            )

        # Associate each restaurant with its most frequent cuisine.
        for restaurant, group in df.groupby(restaurant_col):
            cuisines = (
                group[cuisine_col]
                .str.strip()
                .replace("", pd.NA)
                .dropna()
            )

            modes = cuisines.mode()

            if not modes.empty:
                restaurant_cuisine[str(restaurant)] = str(
                    modes.iloc[0]
                )

    # Save learned customer-order profiles.
    artifact = {
        "model_type": "customer_order_history_baseline",
        "customer_column": customer_col,
        "restaurant_column": restaurant_col,
        "cuisine_column": cuisine_col,
        "customer_restaurant_counts": (
            customer_restaurant_counts
        ),
        "customer_cuisine_preferences": (
            customer_cuisine_preferences
        ),
        "restaurant_popularity": {
            str(k): int(v)
            for k, v in popularity.items()
        },
        "restaurant_cuisine": restaurant_cuisine,
        "training_rows": len(df),
        "unique_customers": int(
            df[customer_col].nunique()
        ),
        "unique_restaurants": int(
            df[restaurant_col].nunique()
        )
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        MODELS_DIR / "previous_order_model.joblib"
    )

    joblib.dump(artifact, output_path)

    print(f"Previous-order model saved: {output_path}")
    print(f"Customers used: {df[customer_col].nunique()}")
    print(f"Restaurants used: {df[restaurant_col].nunique()}")

    return {
        "model": str(output_path),
        "customers": int(df[customer_col].nunique()),
        "restaurants": int(df[restaurant_col].nunique())
    }


def main():
    results = {}
    errors = {}

    try:
        results["restaurant_model"] = train_restaurant_model()
    except Exception as exc:
        errors["restaurant_model"] = str(exc)

    try:
        results["previous_order_model"] = (
            train_previous_order_model()
        )
    except Exception as exc:
        errors["previous_order_model"] = str(exc)

    print("\n========== TRAINING SUMMARY ==========")

    print(
        json.dumps(
            {
                "trained": results,
                "errors": errors
            },
            indent=2
        )
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()