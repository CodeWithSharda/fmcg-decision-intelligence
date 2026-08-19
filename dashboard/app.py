from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="FMCG Decision Intelligence",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

SALES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "fmcg_sales.csv"
)

PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "demand_predictions.csv"
)

HOLDOUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "inventory_holdout_results.csv"
)

OPTIMIZED_SIMULATION_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "best_inventory_policy_simulation.csv"
)

POLICY_TUNING_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "inventory_policy_tuning.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"]
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["date"]
    )

    holdout = pd.read_csv(
        HOLDOUT_PATH
    )

    optimized = pd.read_csv(
        OPTIMIZED_SIMULATION_PATH,
        parse_dates=["date"]
    )

    tuning = pd.read_csv(
        POLICY_TUNING_PATH
    )

    return (
        sales,
        predictions,
        holdout,
        optimized,
        tuning
    )


(
    sales_df,
    predictions_df,
    holdout_df,
    optimized_df,
    tuning_df
) = load_data()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def format_rupees(value):
    value = float(value)

    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"

    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f} L"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f} K"

    return f"₹{value:,.0f}"


def safe_pct_change(new, old):
    if old == 0:
        return 0

    return ((new - old) / old) * 100


# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------

actual = predictions_df["customer_demand"]
predicted = predictions_df["predicted_demand"]

mae = np.mean(
    np.abs(actual - predicted)
)

rmse = np.sqrt(
    np.mean(
        (actual - predicted) ** 2
    )
)

nonzero_mask = actual > 0

mape = np.mean(
    np.abs(
        (
            actual[nonzero_mask]
            - predicted[nonzero_mask]
        )
        / actual[nonzero_mask]
    )
) * 100


# ---------------------------------------------------------
# HOLDOUT POLICY RESULTS
# ---------------------------------------------------------

baseline_row = (
    holdout_df[
        holdout_df["policy"]
        == "Baseline"
    ]
    .iloc[0]
)

optimized_row = (
    holdout_df[
        holdout_df["policy"]
        == "Forecast-Driven"
    ]
    .iloc[0]
)


lost_sales_reduction = (
    (
        baseline_row["lost_sales"]
        - optimized_row["lost_sales"]
    )
    / baseline_row["lost_sales"]
) * 100


stockout_reduction = (
    (
        baseline_row["stockout_rows"]
        - optimized_row["stockout_rows"]
    )
    / baseline_row["stockout_rows"]
) * 100


inventory_reduction = (
    (
        baseline_row["average_inventory"]
        - optimized_row["average_inventory"]
    )
    / baseline_row["average_inventory"]
) * 100


revenue_improvement = safe_pct_change(
    optimized_row["revenue"],
    baseline_row["revenue"]
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("FMCG Decision Intelligence Platform")

st.caption(
    "Machine-learning demand forecasting, "
    "inventory optimisation and business analytics"
)

st.divider()


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

overview_tab, forecast_tab, inventory_tab, decision_tab = st.tabs(
    [
        "Executive Overview",
        "Demand Forecasting",
        "Inventory Optimisation",
        "SKU Decision Centre"
    ]
)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

with overview_tab:
    st.subheader("Business Performance")

    total_revenue = sales_df["revenue"].sum()

    total_units = sales_df["units_sold"].sum()

    total_demand = sales_df[
        "customer_demand"
    ].sum()

    service_level = (
        total_units
        / total_demand
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Historical Revenue",
        format_rupees(total_revenue)
    )

    col2.metric(
        "Units Sold",
        f"{total_units:,}"
    )

    col3.metric(
        "Demand Fulfilment",
        f"{service_level:.2f}%"
    )

    col4.metric(
        "Forecast MAPE",
        f"{mape:.2f}%"
    )

    st.subheader(
        "Forecast-Driven Inventory Impact"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Service Level",
        f"{optimized_row['service_level']:.2f}%",
        delta=(
            f"+"
            f"{optimized_row['service_level'] - baseline_row['service_level']:.2f}"
            " pp"
        )
    )

    col2.metric(
        "Lost Sales",
        f"{int(optimized_row['lost_sales']):,}",
        delta=(
            f"-{lost_sales_reduction:.1f}%"
        ),
        delta_color="inverse"
    )

    col3.metric(
        "Average Inventory",
        f"{optimized_row['average_inventory']:.1f}",
        delta=(
            f"-{inventory_reduction:.1f}%"
        ),
        delta_color="inverse"
    )

    col4.metric(
        "Holdout Revenue",
        format_rupees(
            optimized_row["revenue"]
        ),
        delta=(
            f"+{revenue_improvement:.2f}%"
        )
    )

    st.info(
        "Inventory-policy results are evaluated on an "
        "untouched July 2026 holdout period. Policy "
        "parameters were selected using May–June data."
    )

    # -----------------------------------------------------
    # MONTHLY REVENUE
    # -----------------------------------------------------

    monthly_revenue = (
        sales_df
        .set_index("date")
        .resample("ME")["revenue"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_revenue,
        x="date",
        y="revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (₹)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CITY + CATEGORY
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    revenue_city = (
        sales_df
        .groupby("city", as_index=False)[
            "revenue"
        ]
        .sum()
        .sort_values(
            "revenue",
            ascending=False
        )
    )

    city_fig = px.bar(
        revenue_city,
        x="city",
        y="revenue",
        title="Revenue by City"
    )

    col1.plotly_chart(
        city_fig,
        use_container_width=True
    )

    category_sales = (
        sales_df
        .groupby(
            "category",
            as_index=False
        )["units_sold"]
        .sum()
    )

    category_fig = px.pie(
        category_sales,
        names="category",
        values="units_sold",
        title="Sales Mix by Category"
    )

    col2.plotly_chart(
        category_fig,
        use_container_width=True
    )


# =========================================================
# DEMAND FORECASTING
# =========================================================

with forecast_tab:
    st.subheader(
        "XGBoost Demand Forecasting"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:.2f} units"
    )

    col2.metric(
        "RMSE",
        f"{rmse:.2f} units"
    )

    col3.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

    st.caption(
        "The model predicts customer demand rather than "
        "observed sales so inventory constraints do not "
        "artificially suppress the forecasting target."
    )

    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    cities = sorted(
        predictions_df[
            "city"
        ].unique()
    )

    selected_city = st.selectbox(
        "Select City",
        ["All"] + cities
    )

    filtered_predictions = (
        predictions_df.copy()
    )

    if selected_city != "All":
        filtered_predictions = (
            filtered_predictions[
                filtered_predictions[
                    "city"
                ]
                == selected_city
            ]
        )

    products = sorted(
        filtered_predictions[
            "product_name"
        ].unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        ["All"] + products
    )

    if selected_product != "All":
        filtered_predictions = (
            filtered_predictions[
                filtered_predictions[
                    "product_name"
                ]
                == selected_product
            ]
        )

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    daily_forecast = (
        filtered_predictions
        .groupby(
            "date",
            as_index=False
        )[
            [
                "customer_demand",
                "predicted_demand"
            ]
        ]
        .sum()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_forecast["date"],
            y=daily_forecast[
                "customer_demand"
            ],
            mode="lines",
            name="Actual Demand"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily_forecast["date"],
            y=daily_forecast[
                "predicted_demand"
            ],
            mode="lines",
            name="Predicted Demand"
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Demand",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # ERROR DISTRIBUTION
    # -----------------------------------------------------

    error_fig = px.histogram(
        filtered_predictions,
        x="absolute_error",
        nbins=30,
        title="Forecast Error Distribution"
    )

    error_fig.update_layout(
        xaxis_title="Absolute Error (Units)"
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True
    )


# =========================================================
# INVENTORY OPTIMISATION
# =========================================================

with inventory_tab:
    st.subheader(
        "July Holdout Evaluation"
    )

    st.caption(
        "The inventory policy was tuned on May–June "
        "and locked before evaluating July."
    )

    comparison = holdout_df.copy()

    # -----------------------------------------------------
    # SERVICE LEVEL
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    service_fig = px.bar(
        comparison,
        x="policy",
        y="service_level",
        text_auto=".2f",
        title="Demand Service Level"
    )

    service_fig.update_layout(
        yaxis_title="Service Level (%)"
    )

    col1.plotly_chart(
        service_fig,
        use_container_width=True
    )

    inventory_fig = px.bar(
        comparison,
        x="policy",
        y="average_inventory",
        text_auto=".1f",
        title="Average Ending Inventory"
    )

    inventory_fig.update_layout(
        yaxis_title="Units"
    )

    col2.plotly_chart(
        inventory_fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # BUSINESS IMPACT
    # -----------------------------------------------------

    st.subheader(
        "Measured Holdout Impact"
    )

    impact_df = pd.DataFrame(
        {
            "Metric": [
                "Lost Sales Reduction",
                "Stockout Event Reduction",
                "Average Inventory Reduction",
                "Revenue Improvement"
            ],
            "Improvement (%)": [
                lost_sales_reduction,
                stockout_reduction,
                inventory_reduction,
                revenue_improvement
            ]
        }
    )

    impact_fig = px.bar(
        impact_df,
        x="Metric",
        y="Improvement (%)",
        text_auto=".1f",
        title="Forecast-Driven Policy Improvement"
    )

    st.plotly_chart(
        impact_fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # POLICY TRADE-OFF
    # -----------------------------------------------------

    st.subheader(
        "Policy Sensitivity Analysis"
    )

    policy_fig = px.scatter(
        tuning_df,
        x="average_inventory",
        y="service_level",
        color="order_up_to_days",
        size="service_z",
        hover_data=[
            "lost_sales",
            "stockout_rows",
            "revenue"
        ],
        title=(
            "Inventory vs Service-Level Trade-off"
        )
    )

    policy_fig.add_hline(
        y=97.5,
        line_dash="dash",
        annotation_text=(
            "97.5% service target"
        )
    )

    policy_fig.update_layout(
        xaxis_title="Average Ending Inventory",
        yaxis_title="Service Level (%)"
    )

    st.plotly_chart(
        policy_fig,
        use_container_width=True
    )


# =========================================================
# SKU DECISION CENTRE
# =========================================================

with decision_tab:
    st.subheader(
        "SKU-Level Inventory Decision Centre"
    )

    latest_date = (
        optimized_df["date"].max()
    )

    latest_df = (
        optimized_df[
            optimized_df["date"]
            == latest_date
        ]
        .copy()
    )

    # -----------------------------------------------------
    # DERIVED INVENTORY METRICS
    # -----------------------------------------------------

    # Inventory already committed by suppliers
    latest_df["inventory_on_order"] = (
        latest_df["inventory_position"]
        - latest_df["ending_inventory"]
    ).clip(lower=0)

    # Inventory position after today's newly recommended order
    latest_df["post_order_inventory_position"] = (
        latest_df["inventory_position"]
        + latest_df["order_qty"]
    )

    # Physical inventory cover
    latest_df["physical_cover_days"] = (
        latest_df["ending_inventory"]
        /
        latest_df["predicted_demand"].clip(lower=0.1)
    )

    # Effective cover including stock already on order
    latest_df["inventory_cover_days"] = (
        latest_df["inventory_position"]
        /
        latest_df["predicted_demand"].clip(lower=0.1)
    )

    # Cover after today's replenishment decision
    latest_df["post_order_cover_days"] = (
        latest_df["post_order_inventory_position"]
        /
        latest_df["predicted_demand"].clip(lower=0.1)
    )

    # -----------------------------------------------------
    # OPERATIONAL STATUS
    # -----------------------------------------------------

    latest_df["inventory_status"] = np.select(
        [
            latest_df["stockout_flag"],
            latest_df["order_qty"] > 0,
            latest_df["inventory_on_order"] > 0
        ],
        [
            "STOCKOUT",
            "REORDER DUE",
            "IN TRANSIT"
        ],
        default="HEALTHY"
    )

    # -----------------------------------------------------
    # RECOMMENDED ACTION
    # -----------------------------------------------------

    latest_df["recommended_action"] = np.select(
        [
            latest_df["stockout_flag"],
            latest_df["order_qty"] > 0,
            latest_df["inventory_on_order"] > 0
        ],
        [
            "Expedite supply and replenish",
            "Place replenishment order",
            "Monitor incoming shipment"
        ],
        default="No action needed"
    )

    st.caption(
        f"Latest simulation date: "
        f"{latest_date.date()}"
    )

    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    city_filter = col1.selectbox(
        "Decision Centre City",
        ["All"]
        + sorted(
            latest_df[
                "city"
            ].unique()
        )
    )

    status_filter = col2.selectbox(
        "Inventory Status",
        [
            "All",
            "STOCKOUT",
            "REORDER DUE",
            "IN TRANSIT",
            "HEALTHY"
        ]
    )

    decision_df = latest_df.copy()

    if city_filter != "All":
        decision_df = decision_df[
            decision_df["city"]
            == city_filter
        ]

    if status_filter != "All":
        decision_df = decision_df[
            decision_df[
                "inventory_status"
            ]
            == status_filter
        ]

    # -----------------------------------------------------
    # PRIORITISE ACTIONS
    # -----------------------------------------------------

    priority_map = {
        "STOCKOUT": 1,
        "REORDER DUE": 2,
        "IN TRANSIT": 3,
        "HEALTHY": 4
    }

    decision_df["priority"] = (
        decision_df[
            "inventory_status"
        ]
        .map(priority_map)
    )

    decision_df = (
        decision_df
        .sort_values(
            [
                "priority",
                "order_qty"
            ],
            ascending=[
                True,
                False
            ]
        )
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    stockout_count = int(
        decision_df[
            "stockout_flag"
        ].sum()
    )

    reorder_count = int(
        decision_df[
            "order_qty"
        ]
        .gt(0)
        .sum()
    )

    total_orders = int(
        decision_df[
            "order_qty"
        ].sum()
    )

    if len(decision_df) > 0:
        avg_post_order_cover = (
            decision_df[
                "post_order_cover_days"
            ].mean()
        )
    else:
        avg_post_order_cover = 0.0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Stockouts",
        stockout_count
    )

    col2.metric(
        "SKUs Requiring Orders",
        reorder_count
    )

    col3.metric(
        "Recommended Order Qty",
        f"{total_orders:,}"
    )

    col4.metric(
        "Post-Order Inventory Cover",
        f"{avg_post_order_cover:.1f} days"
    )

    # -----------------------------------------------------
    # DECISION TABLE
    # -----------------------------------------------------

    if len(decision_df) == 0:
        st.info(
            "No store-SKU combinations match "
            "the selected filters."
        )

    else:
        display_df = decision_df[
            [
                "city",
                "product_name",
                "predicted_demand",
                "ending_inventory",
                "inventory_on_order",
                "physical_cover_days",
                "inventory_cover_days",
                "order_qty",
                "post_order_cover_days",
                "inventory_status",
                "recommended_action"
            ]
        ].copy()

        display_df = display_df.rename(
            columns={
                "city":
                    "City",

                "product_name":
                    "Product",

                "predicted_demand":
                    "Predicted Daily Demand",

                "ending_inventory":
                    "On-Hand Inventory",

                "inventory_on_order":
                    "Already On Order",

                "physical_cover_days":
                    "On-Hand Cover (Days)",

                "inventory_cover_days":
                    "Effective Cover (Days)",

                "order_qty":
                    "Recommended Order",

                "post_order_cover_days":
                    "Post-Order Cover (Days)",

                "inventory_status":
                    "Status",

                "recommended_action":
                    "Recommended Action"
            }
        )

        numeric_columns = [
            "Predicted Daily Demand",
            "Already On Order",
            "On-Hand Cover (Days)",
            "Effective Cover (Days)",
            "Post-Order Cover (Days)"
        ]

        display_df[
            numeric_columns
        ] = (
            display_df[
                numeric_columns
            ]
            .round(2)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )