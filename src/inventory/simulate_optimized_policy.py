from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

SALES_PATH = Path("data/processed/fmcg_sales.csv")
PREDICTIONS_PATH = Path(
    "data/processed/demand_predictions.csv"
)

OUTPUT_PATH = Path(
    "data/processed/optimized_inventory_simulation.csv"
)


# ---------------------------------------------------------
# POLICY PARAMETERS
# ---------------------------------------------------------

LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
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


print("\nOPTIMISED INVENTORY POLICY SIMULATION")
print("=" * 55)


# ---------------------------------------------------------
# HISTORICAL DEMAND VARIABILITY
# ---------------------------------------------------------

grouped_demand = sales_df.groupby(
    ["store_id", "sku_id"]
)["customer_demand"]


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


sales_df["demand_std_28"] = (
    sales_df["demand_std_28"]
    .fillna(0)
)


# ---------------------------------------------------------
# MERGE FORECASTS WITH ACTUAL BUSINESS DATA
# ---------------------------------------------------------

context_columns = [
    "date",
    "store_id",
    "sku_id",
    "city",
    "product_name",
    "customer_demand",
    "opening_inventory",
    "ending_inventory",
    "lost_sales",
    "stockout_flag",
    "selling_price",
    "revenue",
    "demand_std_28"
]


simulation_df = predictions_df[
    [
        "date",
        "store_id",
        "sku_id",
        "predicted_demand"
    ]
].merge(
    sales_df[context_columns],
    on=[
        "date",
        "store_id",
        "sku_id"
    ],
    how="left"
)


simulation_df = simulation_df.sort_values(
    ["date", "store_id", "sku_id"]
).reset_index(drop=True)


# ---------------------------------------------------------
# STATE FOR EACH STORE-SKU
# ---------------------------------------------------------

inventory_state = {}
pending_orders = {}

results = []


# ---------------------------------------------------------
# RUN DAY-BY-DAY SIMULATION
# ---------------------------------------------------------

for _, row in simulation_df.iterrows():

    key = (
        row["store_id"],
        row["sku_id"]
    )

    current_date = row["date"]


    # -----------------------------------------------------
    # INITIAL INVENTORY
    # -----------------------------------------------------

    if key not in inventory_state:

        inventory_state[key] = int(
            row["opening_inventory"]
        )

        pending_orders[key] = []


    # -----------------------------------------------------
    # RECEIVE ORDERS THAT HAVE ARRIVED
    # -----------------------------------------------------

    received_qty = 0
    remaining_orders = []

    for order in pending_orders[key]:

        if order["delivery_date"] <= current_date:

            received_qty += order["quantity"]

        else:

            remaining_orders.append(order)


    pending_orders[key] = remaining_orders

    inventory_state[key] += received_qty


    # -----------------------------------------------------
    # OPENING INVENTORY
    # -----------------------------------------------------

    opening_inventory = inventory_state[key]


    # -----------------------------------------------------
    # ACTUAL CUSTOMER DEMAND
    # -----------------------------------------------------

    actual_demand = int(
        row["customer_demand"]
    )


    optimized_units_sold = min(
        actual_demand,
        opening_inventory
    )


    optimized_lost_sales = max(
        actual_demand - opening_inventory,
        0
    )


    ending_inventory = (
        opening_inventory
        - optimized_units_sold
    )


    inventory_state[key] = ending_inventory


    # -----------------------------------------------------
    # FORECAST
    # -----------------------------------------------------

    predicted_demand = max(
        float(row["predicted_demand"]),
        0.1
    )


    # -----------------------------------------------------
    # SAFETY STOCK
    # -----------------------------------------------------

    safety_stock = (
        SERVICE_LEVEL_Z
        * float(row["demand_std_28"])
        * np.sqrt(LEAD_TIME_DAYS)
    )


    # -----------------------------------------------------
    # REORDER POINT
    # -----------------------------------------------------

    lead_time_demand = (
        predicted_demand
        * LEAD_TIME_DAYS
    )


    reorder_point = (
        lead_time_demand
        + safety_stock
    )


    # -----------------------------------------------------
    # TARGET STOCK
    # -----------------------------------------------------

    target_stock = (
        predicted_demand
        * (
            LEAD_TIME_DAYS
            + REVIEW_PERIOD_DAYS
        )
        + safety_stock
    )


    # -----------------------------------------------------
    # INVENTORY POSITION
    # -----------------------------------------------------

    quantity_already_on_order = sum(
        order["quantity"]
        for order in pending_orders[key]
    )


    inventory_position = (
        ending_inventory
        + quantity_already_on_order
    )


    # -----------------------------------------------------
    # PLACE REPLENISHMENT ORDER
    # -----------------------------------------------------

    order_qty = 0

    if inventory_position <= reorder_point:

        order_qty = int(
            np.ceil(
                max(
                    target_stock
                    - inventory_position,
                    0
                )
            )
        )


        if order_qty > 0:

            pending_orders[key].append(
                {
                    "delivery_date":
                        current_date
                        + pd.Timedelta(
                            days=LEAD_TIME_DAYS
                        ),

                    "quantity":
                        order_qty
                }
            )


    # -----------------------------------------------------
    # OPTIMISED REVENUE
    # -----------------------------------------------------

    optimized_revenue = (
        optimized_units_sold
        * float(row["selling_price"])
    )


    # -----------------------------------------------------
    # SAVE ROW
    # -----------------------------------------------------

    results.append(
        {
            "date": current_date,
            "store_id": row["store_id"],
            "city": row["city"],
            "sku_id": row["sku_id"],
            "product_name": row["product_name"],

            "actual_demand":
                actual_demand,

            "predicted_demand":
                round(predicted_demand, 2),

            "received_qty":
                received_qty,

            "opening_inventory":
                opening_inventory,

            "units_sold":
                optimized_units_sold,

            "lost_sales":
                optimized_lost_sales,

            "ending_inventory":
                ending_inventory,

            "safety_stock":
                round(safety_stock, 2),

            "reorder_point":
                round(reorder_point, 2),

            "inventory_position":
                round(inventory_position, 2),

            "order_qty":
                order_qty,

            "optimized_revenue":
                round(optimized_revenue, 2)
        }
    )


# ---------------------------------------------------------
# CREATE RESULTS TABLE
# ---------------------------------------------------------

optimized_df = pd.DataFrame(results)


optimized_df["stockout_flag"] = (
    optimized_df["lost_sales"] > 0
)


optimized_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# FAIR BASELINE COMPARISON
# ---------------------------------------------------------

test_start = optimized_df["date"].min()
test_end = optimized_df["date"].max()


baseline_df = sales_df[
    (sales_df["date"] >= test_start)
    &
    (sales_df["date"] <= test_end)
].copy()


baseline_lost_sales = (
    baseline_df["lost_sales"].sum()
)

optimized_lost_sales = (
    optimized_df["lost_sales"].sum()
)


baseline_stockout_rows = (
    baseline_df["stockout_flag"].sum()
)

optimized_stockout_rows = (
    optimized_df["stockout_flag"].sum()
)


baseline_revenue = (
    baseline_df["revenue"].sum()
)

optimized_revenue = (
    optimized_df["optimized_revenue"].sum()
)


baseline_average_inventory = (
    baseline_df["ending_inventory"].mean()
)

optimized_average_inventory = (
    optimized_df["ending_inventory"].mean()
)


baseline_service_level = (
    baseline_df["units_sold"].sum()
    /
    baseline_df["customer_demand"].sum()
) * 100


optimized_service_level = (
    optimized_df["units_sold"].sum()
    /
    optimized_df["actual_demand"].sum()
) * 100


# ---------------------------------------------------------
# IMPROVEMENTS
# ---------------------------------------------------------

lost_sales_reduction = (
    (
        baseline_lost_sales
        - optimized_lost_sales
    )
    / baseline_lost_sales
) * 100


stockout_reduction = (
    (
        baseline_stockout_rows
        - optimized_stockout_rows
    )
    / baseline_stockout_rows
) * 100


revenue_recovered = (
    optimized_revenue
    - baseline_revenue
)


# ---------------------------------------------------------
# PRINT COMPARISON
# ---------------------------------------------------------

print(
    "\nSimulation period:",
    test_start.date(),
    "to",
    test_end.date()
)


print("\nBASELINE POLICY")
print(
    "Lost sales:",
    f"{baseline_lost_sales:,}"
)

print(
    "Stockout rows:",
    f"{baseline_stockout_rows:,}"
)

print(
    "Service level:",
    f"{baseline_service_level:.2f}%"
)

print(
    "Average ending inventory:",
    f"{baseline_average_inventory:.2f}"
)

print(
    "Revenue:",
    f"₹{baseline_revenue:,.2f}"
)


print("\nOPTIMISED FORECAST-DRIVEN POLICY")

print(
    "Lost sales:",
    f"{optimized_lost_sales:,}"
)

print(
    "Stockout rows:",
    f"{optimized_stockout_rows:,}"
)

print(
    "Service level:",
    f"{optimized_service_level:.2f}%"
)

print(
    "Average ending inventory:",
    f"{optimized_average_inventory:.2f}"
)

print(
    "Revenue:",
    f"₹{optimized_revenue:,.2f}"
)


print("\nBUSINESS IMPACT")

print(
    "Lost-sales reduction:",
    f"{lost_sales_reduction:.2f}%"
)

print(
    "Stockout-event reduction:",
    f"{stockout_reduction:.2f}%"
)

print(
    "Additional revenue captured:",
    f"₹{revenue_recovered:,.2f}"
)


print("\nSimulation saved to:")
print(
    "data/processed/"
    "optimized_inventory_simulation.csv"
)

print(
    "\nOPTIMISED INVENTORY SIMULATION COMPLETE"
)