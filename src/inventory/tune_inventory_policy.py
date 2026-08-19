from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

SALES_PATH = Path(
    "data/processed/fmcg_sales.csv"
)

PREDICTIONS_PATH = Path(
    "data/processed/demand_predictions.csv"
)

TUNING_RESULTS_PATH = Path(
    "data/processed/inventory_policy_tuning.csv"
)

HOLDOUT_RESULTS_PATH = Path(
    "data/processed/inventory_holdout_results.csv"
)

BEST_SIMULATION_PATH = Path(
    "data/processed/best_inventory_policy_simulation.csv"
)

FIGURE_PATH = Path(
    "reports/figures"
)

FIGURE_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# BUSINESS ASSUMPTIONS
# ---------------------------------------------------------

LEAD_TIME_DAYS = 3

TARGET_SERVICE_LEVEL = 97.5


# ---------------------------------------------------------
# POLICIES TO TEST
# ---------------------------------------------------------

SERVICE_Z_VALUES = [
    0.84,
    1.04,
    1.28,
    1.44,
    1.65,
    1.96
]

ORDER_UP_TO_DAYS = [
    5,
    7,
    10,
    14
]


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
    [
        "store_id",
        "sku_id",
        "date"
    ]
).reset_index(drop=True)


print(
    "\nINVENTORY POLICY TUNING + HOLDOUT EVALUATION"
)

print("=" * 60)


# ---------------------------------------------------------
# HISTORICAL DEMAND VOLATILITY
# ---------------------------------------------------------

grouped_demand = sales_df.groupby(
    [
        "store_id",
        "sku_id"
    ]
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
# BUILD SIMULATION DATA
# ---------------------------------------------------------

context_columns = [
    "date",
    "store_id",
    "sku_id",
    "city",
    "product_name",
    "customer_demand",
    "opening_inventory",
    "selling_price",
    "demand_std_28"
]


simulation_data = (
    predictions_df[
        [
            "date",
            "store_id",
            "sku_id",
            "predicted_demand"
        ]
    ]
    .merge(
        sales_df[
            context_columns
        ],
        on=[
            "date",
            "store_id",
            "sku_id"
        ],
        how="left"
    )
    .sort_values(
        [
            "date",
            "store_id",
            "sku_id"
        ]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# TIME SPLIT FOR INVENTORY POLICY
# ---------------------------------------------------------

TUNING_END_DATE = pd.Timestamp(
    "2026-06-30"
)

HOLDOUT_START_DATE = pd.Timestamp(
    "2026-07-01"
)


tuning_data = simulation_data[
    simulation_data["date"]
    <= TUNING_END_DATE
].copy()


holdout_data = simulation_data[
    simulation_data["date"]
    >= HOLDOUT_START_DATE
].copy()


print("\nPOLICY TUNING PERIOD")

print(
    tuning_data["date"].min().date(),
    "to",
    tuning_data["date"].max().date()
)


print("\nUNTOUCHED HOLDOUT PERIOD")

print(
    holdout_data["date"].min().date(),
    "to",
    holdout_data["date"].max().date()
)


# ---------------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------------

def simulate_policy(
    input_data,
    service_z,
    order_up_to_days
):

    inventory_state = {}
    pending_orders = {}

    results = []


    for _, row in input_data.iterrows():

        key = (
            row["store_id"],
            row["sku_id"]
        )

        current_date = row["date"]


        # -------------------------------------------------
        # INITIAL INVENTORY
        # -------------------------------------------------

        if key not in inventory_state:

            inventory_state[key] = int(
                row["opening_inventory"]
            )

            pending_orders[key] = []


        # -------------------------------------------------
        # RECEIVE ARRIVING ORDERS
        # -------------------------------------------------

        received_qty = 0
        remaining_orders = []


        for order in pending_orders[key]:

            if (
                order["delivery_date"]
                <= current_date
            ):

                received_qty += (
                    order["quantity"]
                )

            else:

                remaining_orders.append(
                    order
                )


        pending_orders[key] = (
            remaining_orders
        )


        inventory_state[key] += (
            received_qty
        )


        # -------------------------------------------------
        # OPENING INVENTORY
        # -------------------------------------------------

        opening_inventory = (
            inventory_state[key]
        )


        # -------------------------------------------------
        # ACTUAL DEMAND — USED ONLY FOR EVALUATION
        # -------------------------------------------------

        actual_demand = int(
            row["customer_demand"]
        )


        units_sold = min(
            actual_demand,
            opening_inventory
        )


        lost_sales = max(
            actual_demand
            - opening_inventory,
            0
        )


        ending_inventory = (
            opening_inventory
            - units_sold
        )


        inventory_state[key] = (
            ending_inventory
        )


        # -------------------------------------------------
        # MODEL FORECAST — USED FOR DECISIONS
        # -------------------------------------------------

        predicted_demand = max(
            float(
                row["predicted_demand"]
            ),
            0.1
        )


        demand_std = float(
            row["demand_std_28"]
        )


        # -------------------------------------------------
        # SAFETY STOCK
        # -------------------------------------------------

        safety_stock = (
            service_z
            * demand_std
            * np.sqrt(
                LEAD_TIME_DAYS
            )
        )


        # -------------------------------------------------
        # REORDER POINT
        # -------------------------------------------------

        lead_time_demand = (
            predicted_demand
            * LEAD_TIME_DAYS
        )


        reorder_point = (
            lead_time_demand
            + safety_stock
        )


        # -------------------------------------------------
        # TARGET INVENTORY
        # -------------------------------------------------

        target_stock = (
            predicted_demand
            * order_up_to_days
            + safety_stock
        )


        # -------------------------------------------------
        # INVENTORY POSITION
        # -------------------------------------------------

        quantity_on_order = sum(
            order["quantity"]
            for order
            in pending_orders[key]
        )


        inventory_position = (
            ending_inventory
            + quantity_on_order
        )


        # -------------------------------------------------
        # REPLENISHMENT DECISION
        # -------------------------------------------------

        order_qty = 0


        if (
            inventory_position
            <= reorder_point
        ):

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
                                days=
                                LEAD_TIME_DAYS
                            ),

                        "quantity":
                            order_qty
                    }
                )


        # -------------------------------------------------
        # REVENUE
        # -------------------------------------------------

        revenue = (
            units_sold
            * float(
                row["selling_price"]
            )
        )


        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        results.append(
            {
                "date":
                    current_date,

                "store_id":
                    row["store_id"],

                "city":
                    row["city"],

                "sku_id":
                    row["sku_id"],

                "product_name":
                    row["product_name"],

                "actual_demand":
                    actual_demand,

                "predicted_demand":
                    round(
                        predicted_demand,
                        2
                    ),

                "opening_inventory":
                    opening_inventory,

                "received_qty":
                    received_qty,

                "units_sold":
                    units_sold,

                "lost_sales":
                    lost_sales,

                "ending_inventory":
                    ending_inventory,

                "safety_stock":
                    round(
                        safety_stock,
                        2
                    ),

                "reorder_point":
                    round(
                        reorder_point,
                        2
                    ),

                "inventory_position":
                    round(
                        inventory_position,
                        2
                    ),

                "order_qty":
                    order_qty,

                "revenue":
                    round(
                        revenue,
                        2
                    )
            }
        )


    result_df = pd.DataFrame(
        results
    )


    result_df["stockout_flag"] = (
        result_df["lost_sales"]
        > 0
    )


    # -----------------------------------------------------
    # QUALITY CHECKS
    # -----------------------------------------------------

    assert (
        result_df[
            "ending_inventory"
        ] >= 0
    ).all()

    assert (
        result_df[
            "order_qty"
        ] >= 0
    ).all()


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    total_demand = (
        result_df[
            "actual_demand"
        ].sum()
    )

    total_sales = (
        result_df[
            "units_sold"
        ].sum()
    )


    service_level = (
        total_sales
        / total_demand
    ) * 100


    metrics = {

        "service_z":
            service_z,

        "order_up_to_days":
            order_up_to_days,

        "service_level":
            service_level,

        "lost_sales":
            result_df[
                "lost_sales"
            ].sum(),

        "stockout_rows":
            result_df[
                "stockout_flag"
            ].sum(),

        "average_inventory":
            result_df[
                "ending_inventory"
            ].mean(),

        "revenue":
            result_df[
                "revenue"
            ].sum(),

        "total_ordered":
            result_df[
                "order_qty"
            ].sum()
    }


    return metrics, result_df


# ---------------------------------------------------------
# TUNE POLICIES USING MAY + JUNE ONLY
# ---------------------------------------------------------

policy_results = []


for service_z in SERVICE_Z_VALUES:

    for cover_days in ORDER_UP_TO_DAYS:

        print(
            f"Testing Z={service_z}, "
            f"target={cover_days} days"
        )


        metrics, _ = simulate_policy(
            tuning_data,
            service_z,
            cover_days
        )


        policy_results.append(
            metrics
        )


# ---------------------------------------------------------
# SAVE TUNING RESULTS
# ---------------------------------------------------------

tuning_df = pd.DataFrame(
    policy_results
)


tuning_df.to_csv(
    TUNING_RESULTS_PATH,
    index=False
)


# ---------------------------------------------------------
# SELECT POLICY — USING TUNING PERIOD ONLY
# ---------------------------------------------------------

eligible_policies = tuning_df[
    tuning_df["service_level"]
    >= TARGET_SERVICE_LEVEL
].copy()


if len(eligible_policies) > 0:

    best_policy = (
        eligible_policies
        .sort_values(
            [
                "average_inventory",
                "lost_sales"
            ],
            ascending=[
                True,
                True
            ]
        )
        .iloc[0]
    )

else:

    best_policy = (
        tuning_df
        .sort_values(
            [
                "service_level",
                "average_inventory"
            ],
            ascending=[
                False,
                True
            ]
        )
        .iloc[0]
    )


best_z = float(
    best_policy[
        "service_z"
    ]
)


best_cover = int(
    best_policy[
        "order_up_to_days"
    ]
)


print("\nSELECTED USING TUNING PERIOD ONLY")

print(
    "Safety-stock Z:",
    best_z
)

print(
    "Target coverage:",
    best_cover,
    "days"
)

print(
    "Tuning service level:",
    f"{best_policy['service_level']:.2f}%"
)

print(
    "Tuning average inventory:",
    f"{best_policy['average_inventory']:.2f}"
)


# ---------------------------------------------------------
# IMPORTANT:
# RERUN SELECTED POLICY FROM MAY THROUGH JULY
#
# This allows July to inherit the real inventory position
# created by the selected policy during May and June.
# ---------------------------------------------------------

full_selected_metrics, full_selected_df = (
    simulate_policy(
        simulation_data,
        best_z,
        best_cover
    )
)


full_selected_df.to_csv(
    BEST_SIMULATION_PATH,
    index=False
)


# ---------------------------------------------------------
# JULY HOLDOUT ONLY
# ---------------------------------------------------------

optimized_holdout = full_selected_df[
    full_selected_df["date"]
    >= HOLDOUT_START_DATE
].copy()


baseline_holdout = sales_df[
    (
        sales_df["date"]
        >= HOLDOUT_START_DATE
    )
    &
    (
        sales_df["date"]
        <= optimized_holdout[
            "date"
        ].max()
    )
].copy()


# ---------------------------------------------------------
# HOLDOUT METRICS
# ---------------------------------------------------------

baseline_demand = (
    baseline_holdout[
        "customer_demand"
    ].sum()
)


baseline_units_sold = (
    baseline_holdout[
        "units_sold"
    ].sum()
)


baseline_service_level = (
    baseline_units_sold
    / baseline_demand
) * 100


baseline_lost_sales = (
    baseline_holdout[
        "lost_sales"
    ].sum()
)


baseline_stockout_rows = (
    baseline_holdout[
        "stockout_flag"
    ].sum()
)


baseline_average_inventory = (
    baseline_holdout[
        "ending_inventory"
    ].mean()
)


baseline_revenue = (
    baseline_holdout[
        "revenue"
    ].sum()
)


optimized_demand = (
    optimized_holdout[
        "actual_demand"
    ].sum()
)


optimized_units_sold = (
    optimized_holdout[
        "units_sold"
    ].sum()
)


optimized_service_level = (
    optimized_units_sold
    / optimized_demand
) * 100


optimized_lost_sales = (
    optimized_holdout[
        "lost_sales"
    ].sum()
)


optimized_stockout_rows = (
    optimized_holdout[
        "stockout_flag"
    ].sum()
)


optimized_average_inventory = (
    optimized_holdout[
        "ending_inventory"
    ].mean()
)


optimized_revenue = (
    optimized_holdout[
        "revenue"
    ].sum()
)


# ---------------------------------------------------------
# HOLDOUT BUSINESS IMPACT
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


inventory_change = (
    (
        optimized_average_inventory
        - baseline_average_inventory
    )
    / baseline_average_inventory
) * 100


revenue_change = (
    (
        optimized_revenue
        - baseline_revenue
    )
    / baseline_revenue
) * 100


revenue_recovered = (
    optimized_revenue
    - baseline_revenue
)


# ---------------------------------------------------------
# SAVE HOLDOUT SUMMARY
# ---------------------------------------------------------

holdout_summary = pd.DataFrame(
    [
        {
            "policy":
                "Baseline",

            "service_level":
                baseline_service_level,

            "lost_sales":
                baseline_lost_sales,

            "stockout_rows":
                baseline_stockout_rows,

            "average_inventory":
                baseline_average_inventory,

            "revenue":
                baseline_revenue
        },

        {
            "policy":
                "Forecast-Driven",

            "service_level":
                optimized_service_level,

            "lost_sales":
                optimized_lost_sales,

            "stockout_rows":
                optimized_stockout_rows,

            "average_inventory":
                optimized_average_inventory,

            "revenue":
                optimized_revenue
        }
    ]
)


holdout_summary.to_csv(
    HOLDOUT_RESULTS_PATH,
    index=False
)


# ---------------------------------------------------------
# TUNING TRADE-OFF CHART
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


for cover_days in ORDER_UP_TO_DAYS:

    subset = tuning_df[
        tuning_df[
            "order_up_to_days"
        ]
        == cover_days
    ]


    plt.scatter(
        subset[
            "average_inventory"
        ],
        subset[
            "service_level"
        ],
        label=
            f"{cover_days}-day target"
    )


plt.scatter(
    [
        best_policy[
            "average_inventory"
        ]
    ],
    [
        best_policy[
            "service_level"
        ]
    ],
    marker="*",
    s=180,
    label="Selected Policy"
)


plt.axhline(
    y=TARGET_SERVICE_LEVEL,
    linestyle="--",
    label=
        f"{TARGET_SERVICE_LEVEL}% target"
)


plt.title(
    "Inventory Policy Tuning Trade-off"
)

plt.xlabel(
    "Average Ending Inventory"
)

plt.ylabel(
    "Demand Service Level (%)"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    FIGURE_PATH
    / "inventory_policy_tradeoff.png"
)


plt.close()


# ---------------------------------------------------------
# HOLDOUT COMPARISON CHART
# ---------------------------------------------------------

comparison_df = pd.DataFrame(
    {
        "Baseline": [
            baseline_service_level,
            baseline_average_inventory
        ],

        "Forecast-Driven": [
            optimized_service_level,
            optimized_average_inventory
        ]
    },
    index=[
        "Service Level (%)",
        "Average Inventory"
    ]
)


comparison_df.to_csv(
    "data/processed/"
    "holdout_policy_comparison.csv"
)


# ---------------------------------------------------------
# DISPLAY FINAL RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)

print("FINAL JULY HOLDOUT EVALUATION")

print("=" * 60)


print(
    "\nHoldout period:",
    optimized_holdout[
        "date"
    ].min().date(),
    "to",
    optimized_holdout[
        "date"
    ].max().date()
)


print("\nBASELINE POLICY")

print(
    "Service level:",
    f"{baseline_service_level:.2f}%"
)

print(
    "Lost sales:",
    f"{baseline_lost_sales:,}"
)

print(
    "Stockout rows:",
    f"{baseline_stockout_rows:,}"
)

print(
    "Average ending inventory:",
    f"{baseline_average_inventory:.2f}"
)

print(
    "Revenue:",
    f"₹{baseline_revenue:,.2f}"
)


print(
    "\nFORECAST-DRIVEN POLICY"
)

print(
    "Service level:",
    f"{optimized_service_level:.2f}%"
)

print(
    "Lost sales:",
    f"{optimized_lost_sales:,}"
)

print(
    "Stockout rows:",
    f"{optimized_stockout_rows:,}"
)

print(
    "Average ending inventory:",
    f"{optimized_average_inventory:.2f}"
)

print(
    "Revenue:",
    f"₹{optimized_revenue:,.2f}"
)


print("\nHOLDOUT BUSINESS IMPACT")

print(
    "Lost-sales reduction:",
    f"{lost_sales_reduction:.2f}%"
)

print(
    "Stockout-event reduction:",
    f"{stockout_reduction:.2f}%"
)

print(
    "Average inventory change:",
    f"{inventory_change:.2f}%"
)

print(
    "Revenue improvement:",
    f"{revenue_change:.2f}%"
)

print(
    "Additional revenue captured:",
    f"₹{revenue_recovered:,.2f}"
)


print("\nOutputs saved:")

print(
    "data/processed/"
    "inventory_policy_tuning.csv"
)

print(
    "data/processed/"
    "inventory_holdout_results.csv"
)

print(
    "data/processed/"
    "best_inventory_policy_simulation.csv"
)

print(
    "reports/figures/"
    "inventory_policy_tradeoff.png"
)


print(
    "\nHOLDOUT EVALUATION COMPLETE"
)