# Food Ordering & Delivery ML Project

## Run order
Open Jupyter and run notebooks in `notebooks/` in this order:
00_START_HERE -> 01_load_and_eda -> 02_cleaning_and_preprocessing ->
03_delivery_feature_engineering -> 04_delivery_model_training ->
05_delivery_model_evaluation -> 06_restaurant_recommendations ->
07_predict_unlabelled_test.

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

## Included data
- `data/raw/delivery_time_train.csv`: labelled delivery data.
- `data/raw/delivery_time_test.csv`: unlabelled test data.
- `data/raw/delivery_time_sample_submission.csv`: submission template.
- `data/raw/online_food_orders.csv`: order-level analysis.
- `data/raw/swiggy_restaurant_listings.csv`: restaurant listing/recommendation.

## Important limitations
- No live GPS/traffic feed; ETA model is based on historical records.
- Restaurant recommendations are a static ranking baseline, not a fully personalized recommender.
- The datasets do not have verified shared IDs; do not merge them blindly.
- Test predictions are not model evaluation because true labels are not present.
