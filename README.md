# Food Ordering & Delivery ML Project

## Project Overview

The Food Ordering & Delivery ML Project is a machine learning-based application that predicts food delivery time and recommends restaurants based on ratings and customer order history.

The project combines machine learning, data preprocessing, feature engineering, recommendation techniques, and FastAPI to provide prediction and recommendation services through REST APIs.

## Project Objectives

- Predict estimated food delivery time in minutes.
- Compare Random Forest and Extra Trees regression algorithms.
- Recommend highly rated restaurants using weighted ratings.
- Suggest restaurants based on previous customer orders.
- Expose predictions and recommendations through REST APIs.
- Deploy the application using FastAPI and Render.

##  Key Features

### 1. Delivery Time Prediction
Predicts the estimated delivery time of a food order using available delivery-related features.

Algorithms:
- Random Forest Regressor
- Extra Trees Regressor

Evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 2. Restaurant Recommendations
Ranks restaurants using a Bayesian-style weighted rating formula that considers average ratings and review counts.

Formula:

WR = (v / (v + m)) × R + (m / (v + m)) × C

Where:
- WR = Weighted rating
- R = Restaurant's average rating
- v = Number of restaurant reviews
- m = Minimum review-count threshold
- C = Overall average rating

This approach reduces the influence of high ratings based on very few reviews.

### 3. Previous-Order Recommendations
Uses customer order history to identify frequently ordered cuisines and recommend restaurants matching those preferences.

This feature uses rule-based recommendation logic rather than a separately trained ML model.

##  Machine Learning Workflow

1. Dataset collection and inspection
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Feature engineering
5. Training and testing data preparation
6. Model training
7. Model evaluation and comparison
8. Saving the selected model
9. FastAPI integration
10. Deployment

##  Algorithms Used

### Random Forest Regressor
Random Forest trains multiple decision trees using bootstrap samples by default. It introduces randomness through feature selection and averages the predictions of individual trees to produce the final prediction.

### Extra Trees Regressor
Extra Trees, or Extremely Randomized Trees, builds multiple decision trees with additional randomness in split-threshold selection. It averages the predictions of the trees to generate the final result.

### Median Baseline
A simple baseline predicts the median delivery time from the training data. It provides a reference point for evaluating whether more complex models improve prediction performance.

## Model Evaluation

`Mean Absolute Error (MAE)`

Measures the average absolute difference between actual and predicted delivery times.

MAE = (1/n) × Σ |yᵢ - ŷᵢ|

`Root Mean Squared Error (RMSE)`

Penalizes larger prediction errors more heavily.

RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]

`R² Score`

Measures how well the model explains variation in the target values relative to a mean-prediction baseline.

Models should be evaluated on the same held-out test data. Actual evaluation results should be added after training.

