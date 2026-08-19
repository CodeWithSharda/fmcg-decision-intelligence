from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from xgboost import XGBRegressor


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_PATH = Path("data/processed/fmcg_sales.csv")
MODEL_PATH = Path("models")
FIGURE_PATH = Path("reports/figures")

MODEL_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

sales_df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

sales_df = sales_df.sort_values(
    ["store_id", "sku_id", "date"]
).reset_index(drop=True)


print("\nDEMAND FORECASTING MODEL")
print("=" * 45)

print("Rows loaded:", len(sales_df))


# ---------------------------------------------------------
# DATE FEATURES
# ---------------------------------------------------------

sales_df["year"] = sales_df["date"].dt.year
sales_df["month"] = sales_df["date"].dt.month
sales_df["day"] = sales_df["date"].dt.day
sales_df["day_of_week_num"] = sales_df["date"].dt.dayofweek
sales_df["week_of_year"] = sales_df["date"].dt.isocalendar().week.astype(int)
sales_df["quarter"] = sales_df["date"].dt.quarter


# ---------------------------------------------------------
# LAG FEATURES
# ---------------------------------------------------------

grouped_demand = sales_df.groupby(
    ["store_id", "sku_id"]
)["customer_demand"]


sales_df["lag_1"] = grouped_demand.shift(1)
sales_df["lag_7"] = grouped_demand.shift(7)
sales_df["lag_14"] = grouped_demand.shift(14)
sales_df["lag_28"] = grouped_demand.shift(28)


# ---------------------------------------------------------
# ROLLING DEMAND FEATURES
# ---------------------------------------------------------

sales_df["rolling_mean_7"] = (
    grouped_demand
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

sales_df["rolling_mean_28"] = (
    grouped_demand
    .transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
)

sales_df["rolling_std_7"] = (
    grouped_demand
    .transform(
        lambda x: x.shift(1).rolling(7).std()
    )
)


# ---------------------------------------------------------
# REMOVE ROWS WITHOUT ENOUGH HISTORY
# ---------------------------------------------------------

sales_df = sales_df.dropna().reset_index(drop=True)

print("Rows after feature engineering:", len(sales_df))


# ---------------------------------------------------------
# TARGET
# ---------------------------------------------------------

TARGET = "customer_demand"


# ---------------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------------

feature_columns = [
    "store_id",
    "city",
    "sku_id",
    "category",
    "base_price",
    "promotion",
    "discount_pct",
    "selling_price",
    "is_weekend",
    "year",
    "month",
    "day",
    "day_of_week_num",
    "week_of_year",
    "quarter",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7"
]

X = sales_df[feature_columns].copy()
y = sales_df[TARGET].copy()


# ---------------------------------------------------------
# CONVERT BOOLEAN COLUMNS
# ---------------------------------------------------------

X["promotion"] = X["promotion"].astype(int)
X["is_weekend"] = X["is_weekend"].astype(int)


# ---------------------------------------------------------
# ONE-HOT ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=[
        "store_id",
        "city",
        "sku_id",
        "category"
    ],
    drop_first=False
)


# ---------------------------------------------------------
# TIME-BASED TRAIN / TEST SPLIT
# ---------------------------------------------------------

split_date = sales_df["date"].max() - pd.Timedelta(days=90)

train_mask = sales_df["date"] < split_date
test_mask = sales_df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

test_dates = sales_df.loc[test_mask, "date"]


print("\nTRAIN / TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Test period begins:", split_date.date())


# ---------------------------------------------------------
# BASELINE MODEL
# ---------------------------------------------------------

baseline_predictions = sales_df.loc[test_mask, "lag_7"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)


print("\nBASELINE MODEL")
print(f"MAE: {baseline_mae:.3f}")
print(f"RMSE: {baseline_rmse:.3f}")


# ---------------------------------------------------------
# XGBOOST MODEL
# ---------------------------------------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

predictions = model.predict(X_test)

# Demand cannot be negative
predictions = np.maximum(predictions, 0)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

nonzero_mask = y_test > 0

mape = np.mean(
    np.abs(
        (
            y_test[nonzero_mask]
            - predictions[nonzero_mask]
        )
        / y_test[nonzero_mask]
    )
) * 100


print("\nXGBOOST PERFORMANCE")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.2f}%")


# ---------------------------------------------------------
# IMPROVEMENT OVER BASELINE
# ---------------------------------------------------------

mae_improvement = (
    (baseline_mae - mae)
    / baseline_mae
) * 100

print(
    f"MAE improvement over baseline: "
    f"{mae_improvement:.2f}%"
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH / "demand_forecast_xgb.pkl"
)

joblib.dump(
    list(X.columns),
    MODEL_PATH / "model_features.pkl"
)

print("\nModel saved to models/demand_forecast_xgb.pkl")


# ---------------------------------------------------------
# SAVE TEST PREDICTIONS
# ---------------------------------------------------------

prediction_df = sales_df.loc[
    test_mask,
    [
        "date",
        "store_id",
        "city",
        "sku_id",
        "product_name",
        "customer_demand"
    ]
].copy()

prediction_df["predicted_demand"] = predictions

prediction_df["absolute_error"] = np.abs(
    prediction_df["customer_demand"]
    - prediction_df["predicted_demand"]
)

prediction_df.to_csv(
    "data/processed/demand_predictions.csv",
    index=False
)


# ---------------------------------------------------------
# ACTUAL VS PREDICTED CHART
# ---------------------------------------------------------

daily_comparison = (
    prediction_df
    .groupby("date")[
        ["customer_demand", "predicted_demand"]
    ]
    .sum()
)

plt.figure(figsize=(12, 5))

plt.plot(
    daily_comparison.index,
    daily_comparison["customer_demand"],
    label="Actual Demand"
)

plt.plot(
    daily_comparison.index,
    daily_comparison["predicted_demand"],
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Daily Demand")
plt.xlabel("Date")
plt.ylabel("Units")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "actual_vs_predicted_demand.png"
)

plt.close()


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nTOP 10 MODEL FEATURES")
print(importance_df.head(10))


top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.title("Top Demand Forecasting Features")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "feature_importance.png"
)

plt.close()


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

print("\nMODEL TRAINING COMPLETE")
print("Predictions saved to:")
print("data/processed/demand_predictions.csv")

print("\nCharts created:")
print("reports/figures/actual_vs_predicted_demand.png")
print("reports/figures/feature_importance.png")