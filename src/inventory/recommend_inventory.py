from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

SALES_PATH = Path("data/processed/fmcg_sales.csv")
PREDICTIONS_PATH = Path("data/processed/demand_predictions.csv")

OUTPUT_PATH = Path(
    "data/processed/inventory_recommendations.csv"
)


# ---------------------------------------------------------
# INVENTORY POLICY SETTINGS
# ---------------------------------------------------------

# Assume suppliers take approximately 3 days to deliver stock
LEAD_TIME_DAYS = 3

# Inventory is formally reviewed once every 7 days
REVIEW_PERIOD_DAYS = 7

# 1.65 corresponds roughly to a 95% service-level assumption
SERVICE_LEVEL_Z = 1.65


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

sales_df = pd.read_csv(
    SALES_PATH,
    parse_dates=["date"]
)

predictions_df = pd.read_csv(
    PREDICTIONS_PATH,
    parse_dates=["date"]
)


sales_df = sales_df.sort_values(
    ["store_id", "sku_id", "date"]
).reset_index(drop=True)


print("\nFORECAST-DRIVEN INVENTORY RECOMMENDATION ENGINE")
print("=" * 55)


# ---------------------------------------------------------
# HISTORICAL DEMAND VARIABILITY
# ---------------------------------------------------------

grouped_demand = sales_df.groupby(
    ["store_id", "sku_id"]
)["customer_demand"]


sales_df["demand_mean_28"] = (
    grouped_demand
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=28,
            min_periods=7
        )
        .mean()
    )
)


sales_df["demand_std_28"] = (
    grouped_demand
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=28,
            min_periods=7
        )
        .std()
    )
)


# ---------------------------------------------------------
# ADD INVENTORY INFORMATION TO FORECASTS
# ---------------------------------------------------------

inventory_columns = [
    "date",
    "store_id",
    "sku_id",
    "category",
    "ending_inventory",
    "demand_mean_28",
    "demand_std_28",
    "base_price"
]


inventory_context = sales_df[
    inventory_columns
].copy()


recommendations_df = predictions_df.merge(
    inventory_context,
    on=[
        "date",
        "store_id",
        "sku_id"
    ],
    how="left"
)


# ---------------------------------------------------------
# CLEAN / VALIDATE
# ---------------------------------------------------------

recommendations_df["demand_std_28"] = (
    recommendations_df["demand_std_28"]
    .fillna(0)
)

recommendations_df["predicted_demand"] = (
    recommendations_df["predicted_demand"]
    .clip(lower=0.1)
)


# ---------------------------------------------------------
# SAFETY STOCK
# ---------------------------------------------------------

# Safety stock increases when demand is more volatile
# or supplier lead time is longer.

recommendations_df["safety_stock"] = (
    SERVICE_LEVEL_Z
    * recommendations_df["demand_std_28"]
    * np.sqrt(LEAD_TIME_DAYS)
)


# ---------------------------------------------------------
# EXPECTED DEMAND DURING SUPPLIER LEAD TIME
# ---------------------------------------------------------

recommendations_df["lead_time_demand"] = (
    recommendations_df["predicted_demand"]
    * LEAD_TIME_DAYS
)


# ---------------------------------------------------------
# REORDER POINT
# ---------------------------------------------------------

# Reorder Point =
# expected demand while waiting for supplier
# + safety stock

recommendations_df["reorder_point"] = (
    recommendations_df["lead_time_demand"]
    + recommendations_df["safety_stock"]
)


# ---------------------------------------------------------
# TARGET STOCK
# ---------------------------------------------------------

# We want enough inventory to cover:
#
# supplier lead time
# +
# next inventory review period
# +
# safety stock

recommendations_df["target_stock"] = (
    recommendations_df["predicted_demand"]
    * (
        LEAD_TIME_DAYS
        + REVIEW_PERIOD_DAYS
    )
    + recommendations_df["safety_stock"]
)


# ---------------------------------------------------------
# INVENTORY COVER
# ---------------------------------------------------------

recommendations_df["inventory_cover_days"] = (
    recommendations_df["ending_inventory"]
    / recommendations_df["predicted_demand"]
)


# ---------------------------------------------------------
# RECOMMENDED ORDER QUANTITY
# ---------------------------------------------------------

recommendations_df["recommended_order_qty"] = (
    recommendations_df["target_stock"]
    - recommendations_df["ending_inventory"]
)


recommendations_df["recommended_order_qty"] = (
    recommendations_df[
        "recommended_order_qty"
    ]
    .clip(lower=0)
    .apply(np.ceil)
    .astype(int)
)


# ---------------------------------------------------------
# STOCK RISK CLASSIFICATION
# ---------------------------------------------------------

high_risk = (
    recommendations_df["ending_inventory"]
    < recommendations_df["lead_time_demand"]
)

medium_risk = (
    recommendations_df["ending_inventory"]
    < recommendations_df["reorder_point"]
)


recommendations_df["risk_level"] = np.select(
    [
        high_risk,
        medium_risk
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)


# ---------------------------------------------------------
# BUSINESS ACTION
# ---------------------------------------------------------

recommendations_df["recommended_action"] = np.select(
    [
        recommendations_df["risk_level"] == "HIGH",
        recommendations_df["risk_level"] == "MEDIUM",
        recommendations_df["recommended_order_qty"] > 0
    ],
    [
        "Order immediately",
        "Reorder this cycle",
        "Planned replenishment"
    ],
    default="No order needed"
)


# ---------------------------------------------------------
# USE THE MOST RECENT SIMULATED DATE
# ---------------------------------------------------------

latest_date = recommendations_df["date"].max()

latest_recommendations = recommendations_df[
    recommendations_df["date"] == latest_date
].copy()


# ---------------------------------------------------------
# ROUND DISPLAY VALUES
# ---------------------------------------------------------

columns_to_round = [
    "predicted_demand",
    "demand_mean_28",
    "demand_std_28",
    "safety_stock",
    "lead_time_demand",
    "reorder_point",
    "target_stock",
    "inventory_cover_days"
]


latest_recommendations[
    columns_to_round
] = latest_recommendations[
    columns_to_round
].round(2)


# ---------------------------------------------------------
# SORT BY BUSINESS PRIORITY
# ---------------------------------------------------------

risk_priority = {
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3
}


latest_recommendations["risk_priority"] = (
    latest_recommendations["risk_level"]
    .map(risk_priority)
)


latest_recommendations = (
    latest_recommendations
    .sort_values(
        [
            "risk_priority",
            "recommended_order_qty"
        ],
        ascending=[
            True,
            False
        ]
    )
    .drop(
        columns=["risk_priority"]
    )
)


# ---------------------------------------------------------
# QUALITY CHECKS
# ---------------------------------------------------------

assert (
    latest_recommendations[
        "recommended_order_qty"
    ] >= 0
).all()

assert (
    latest_recommendations[
        "ending_inventory"
    ] >= 0
).all()

assert latest_recommendations[
    "predicted_demand"
].notna().all()


# ---------------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------------

latest_recommendations.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# BUSINESS SUMMARY
# ---------------------------------------------------------

print("\nRecommendation date:", latest_date.date())

print(
    "Store-SKU combinations:",
    len(latest_recommendations)
)

print("\nRISK DISTRIBUTION")

print(
    latest_recommendations[
        "risk_level"
    ].value_counts()
)


print(
    "\nTotal recommended replenishment:",
    f"{latest_recommendations['recommended_order_qty'].sum():,}",
    "units"
)


print("\nTOP INVENTORY ACTIONS")

display_columns = [
    "city",
    "product_name",
    "predicted_demand",
    "ending_inventory",
    "inventory_cover_days",
    "safety_stock",
    "reorder_point",
    "recommended_order_qty",
    "risk_level",
    "recommended_action"
]


print(
    latest_recommendations[
        display_columns
    ].head(10).to_string(index=False)
)


print("\nRecommendations saved to:")
print(
    "data/processed/inventory_recommendations.csv"
)

print("\nINVENTORY RECOMMENDATION ENGINE COMPLETE")