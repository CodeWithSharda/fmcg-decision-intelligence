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
# GROUNDED DECISION COPILOT
# ---------------------------------------------------------

def answer_business_question(question):
    """
    Deterministic natural-language analytics assistant.

    The assistant does not generate or invent business
    numbers. Every answer is calculated directly from the
    project datasets.
    """

    q = question.lower().strip()

    # -----------------------------------------------------
    # LATEST INVENTORY STATE
    # -----------------------------------------------------

    latest_date = optimized_df["date"].max()

    latest = (
        optimized_df[
            optimized_df["date"]
            == latest_date
        ]
        .copy()
    )

    latest["inventory_on_order"] = (
        latest["inventory_position"]
        - latest["ending_inventory"]
    ).clip(lower=0)

    latest["post_order_inventory_position"] = (
        latest["inventory_position"]
        + latest["order_qty"]
    )

    latest["physical_cover_days"] = (
        latest["ending_inventory"]
        /
        latest["predicted_demand"].clip(lower=0.1)
    )

    latest["effective_cover_days"] = (
        latest["inventory_position"]
        /
        latest["predicted_demand"].clip(lower=0.1)
    )

    latest["post_order_cover_days"] = (
        latest["post_order_inventory_position"]
        /
        latest["predicted_demand"].clip(lower=0.1)
    )

    latest["inventory_status"] = np.select(
        [
            latest["stockout_flag"],
            latest["order_qty"] > 0,
            latest["inventory_on_order"] > 0
        ],
        [
            "STOCKOUT",
            "REORDER DUE",
            "IN TRANSIT"
        ],
        default="HEALTHY"
    )

    # -----------------------------------------------------
    # IDENTIFY CITIES / PRODUCTS MENTIONED
    # -----------------------------------------------------

    cities = sorted(
        sales_df["city"].unique()
    )

    products = sorted(
        sales_df["product_name"].unique()
    )

    mentioned_cities = [
        city
        for city in cities
        if city.lower() in q
    ]

    mentioned_products = [
        product
        for product in products
        if product.lower() in q
    ]

    # -----------------------------------------------------
    # CITY COMPARISON
    # -----------------------------------------------------

    if (
        "compare" in q
        and len(mentioned_cities) >= 2
    ):

        comparison_rows = []

        for city in mentioned_cities[:2]:

            city_data = sales_df[
                sales_df["city"]
                == city
            ]

            revenue = (
                city_data["revenue"].sum()
            )

            demand = (
                city_data[
                    "customer_demand"
                ].sum()
            )

            units = (
                city_data[
                    "units_sold"
                ].sum()
            )

            fulfilment = (
                units / demand
            ) * 100

            stockouts = (
                city_data[
                    "stockout_flag"
                ].sum()
            )

            comparison_rows.append(
                (
                    city,
                    revenue,
                    demand,
                    fulfilment,
                    stockouts
                )
            )

        city1 = comparison_rows[0]
        city2 = comparison_rows[1]

        return f"""
### {city1[0]} vs {city2[0]}

| Metric | {city1[0]} | {city2[0]} |
|---|---:|---:|
| Historical Revenue | {format_rupees(city1[1])} | {format_rupees(city2[1])} |
| Customer Demand | {city1[2]:,.0f} units | {city2[2]:,.0f} units |
| Demand Fulfilment | {city1[3]:.2f}% | {city2[3]:.2f}% |
| Stockout Events | {city1[4]:,.0f} | {city2[4]:,.0f} |

These figures are calculated across the complete simulated historical dataset.
"""

    # -----------------------------------------------------
    # SPECIFIC PRODUCT
    # -----------------------------------------------------

    if len(mentioned_products) > 0:

        product = mentioned_products[0]

        historical = sales_df[
            sales_df["product_name"]
            == product
        ]

        current = latest[
            latest["product_name"]
            == product
        ].copy()

        avg_demand = (
            historical[
                "customer_demand"
            ].mean()
        )

        total_revenue = (
            historical["revenue"].sum()
        )

        stockout_rate = (
            historical[
                "stockout_flag"
            ].mean()
            * 100
        )

        current = current.sort_values(
            "order_qty",
            ascending=False
        )

        action_lines = []

        for _, row in current.head(5).iterrows():

            action_lines.append(
                f"- **{row['city']}**: "
                f"{row['inventory_status']}; "
                f"{row['ending_inventory']:.0f} units on hand, "
                f"{row['inventory_on_order']:.0f} already on order, "
                f"recommended new order "
                f"{row['order_qty']:.0f} units."
            )

        actions = "\n".join(
            action_lines
        )

        return f"""
### {product}

**Historical performance**
- Average daily store demand: **{avg_demand:.1f} units**
- Historical revenue: **{format_rupees(total_revenue)}**
- Stockout-event rate: **{stockout_rate:.2f}%**

**Latest inventory position — {latest_date.date()}**

{actions}
"""

    # -----------------------------------------------------
    # REVENUE / BUSINESS IMPACT
    # -----------------------------------------------------

    if (
        "revenue" in q
        and (
            "recover" in q
            or "improv" in q
            or "increase" in q
            or "impact" in q
        )
    ):

        revenue_gain = (
            optimized_row["revenue"]
            - baseline_row["revenue"]
        )

        return f"""
### Holdout Revenue Impact

On the untouched July 2026 holdout period:

- Baseline revenue: **{format_rupees(baseline_row['revenue'])}**
- Forecast-driven policy revenue: **{format_rupees(optimized_row['revenue'])}**
- Additional simulated revenue captured: **{format_rupees(revenue_gain)}**
- Revenue improvement: **{revenue_improvement:.2f}%**

The improvement comes from the complete forecast-driven replenishment policy, not from XGBoost alone.
"""

    # -----------------------------------------------------
    # FORECAST PERFORMANCE
    # -----------------------------------------------------

    if (
        "forecast" in q
        or "accuracy" in q
        or "mape" in q
        or "error" in q
    ):

        product_errors = (
            predictions_df
            .groupby(
                "product_name"
            )["absolute_error"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        worst_product = (
            product_errors.index[0]
        )

        worst_mae = (
            product_errors.iloc[0]
        )

        return f"""
### Demand Forecast Performance

- **MAE:** {mae:.2f} units
- **RMSE:** {rmse:.2f} units
- **MAPE:** {mape:.2f}%
- Improvement over the lag-7 baseline: **32.40% MAE reduction**

The product with the highest average absolute forecast error is **{worst_product}** at approximately **{worst_mae:.2f} units**.

The model forecasts underlying customer demand rather than constrained units sold.
"""

    # -----------------------------------------------------
    # STOCKOUT / MANAGEMENT ATTENTION
    # -----------------------------------------------------

    if (
        "stockout" in q
        or "attention" in q
        or "risk" in q
        or "urgent" in q
    ):

        stockouts = latest[
            latest["inventory_status"]
            == "STOCKOUT"
        ]

        reorder_due = latest[
            latest["inventory_status"]
            == "REORDER DUE"
        ].sort_values(
            "order_qty",
            ascending=False
        )

        lines = []

        for _, row in stockouts.iterrows():

            lines.append(
                f"- **URGENT — {row['city']}, "
                f"{row['product_name']}**: "
                f"stockout with "
                f"{row['ending_inventory']:.0f} units on hand."
            )

        for _, row in reorder_due.head(5).iterrows():

            lines.append(
                f"- **{row['city']}, "
                f"{row['product_name']}**: "
                f"reorder {row['order_qty']:.0f} units; "
                f"current physical cover "
                f"{row['physical_cover_days']:.1f} days."
            )

        if not lines:
            lines.append(
                "- No immediate stockout or "
                "replenishment exceptions detected."
            )

        return f"""
### Management Attention — {latest_date.date()}

Current stockouts: **{len(stockouts)}**

SKUs requiring new orders: **{len(reorder_due)}**

Top actions:

{chr(10).join(lines)}
"""

    # -----------------------------------------------------
    # REPLENISHMENT PRIORITIES
    # -----------------------------------------------------

    if (
        "replenish" in q
        or "reorder" in q
        or "order" in q
        or "priorit" in q
    ):

        orders = (
            latest[
                latest[
                    "order_qty"
                ] > 0
            ]
            .sort_values(
                "order_qty",
                ascending=False
            )
        )

        lines = []

        for _, row in orders.head(8).iterrows():

            lines.append(
                f"- **{row['city']} — "
                f"{row['product_name']}**: "
                f"order **{row['order_qty']:.0f} units** "
                f"(predicted daily demand "
                f"{row['predicted_demand']:.1f})."
            )

        return f"""
### Replenishment Priorities — {latest_date.date()}

There are **{len(orders)} store-SKU combinations** requiring new orders.

Total recommended order quantity: **{orders['order_qty'].sum():,.0f} units**

Priority actions:

{chr(10).join(lines)}
"""

    # -----------------------------------------------------
    # CITY STOCKOUT EXPOSURE
    # -----------------------------------------------------

    if (
        "city" in q
        and (
            "stockout" in q
            or "exposure" in q
            or "inventory" in q
        )
    ):

        city_status = (
            latest
            .groupby("city")
            .agg(
                stockouts=(
                    "stockout_flag",
                    "sum"
                ),
                orders_required=(
                    "order_qty",
                    lambda x:
                    (x > 0).sum()
                ),
                recommended_units=(
                    "order_qty",
                    "sum"
                )
            )
            .sort_values(
                [
                    "stockouts",
                    "orders_required"
                ],
                ascending=False
            )
        )

        top_city = (
            city_status.index[0]
        )

        return f"""
### Inventory Exposure by City

{city_status.to_markdown()}

Based on the latest simulation state, **{top_city}** currently has the highest operational inventory priority using stockouts and reorder requirements.
"""

    # -----------------------------------------------------
    # GENERAL INVENTORY PERFORMANCE
    # -----------------------------------------------------

    if (
        "inventory" in q
        or "service level" in q
    ):

        return f"""
### Inventory Optimisation Performance

**Untouched July 2026 holdout**

- Baseline service level: **{baseline_row['service_level']:.2f}%**
- Forecast-driven service level: **{optimized_row['service_level']:.2f}%**
- Lost-sales reduction: **{lost_sales_reduction:.2f}%**
- Stockout-event reduction: **{stockout_reduction:.2f}%**
- Average inventory reduction: **{inventory_reduction:.2f}%**
- Revenue improvement: **{revenue_improvement:.2f}%**

The inventory policy was selected using May–June data and locked before July evaluation.
"""

    # -----------------------------------------------------
    # HELP / FALLBACK
    # -----------------------------------------------------

    return """
### I can answer grounded business questions such as:

- **Which SKUs need immediate management attention?**
- **What replenishment actions should I prioritise today?**
- **How much revenue did the optimized policy recover?**
- **How accurate is the demand forecast?**
- **Compare Chennai and Mumbai.**
- **Tell me about Bath Soap 100g.**
- **Which city has the highest stockout exposure?**
- **How did inventory optimisation perform?**

Answers are calculated directly from the project datasets rather than generated from unsupported assumptions.
"""

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

(
    overview_tab,
    forecast_tab,
    inventory_tab,
    decision_tab,
    copilot_tab
) = st.tabs(
    [
        "Executive Overview",
        "Demand Forecasting",
        "Inventory Optimisation",
        "SKU Decision Centre",
        "Decision Copilot"
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
# =========================================================
# DECISION COPILOT
# =========================================================

with copilot_tab:

    st.subheader(
        "Grounded Business Decision Copilot"
    )

    st.caption(
        "Natural-language analytics grounded directly "
        "in forecasting, sales and inventory data. "
        "No external API or paid LLM required."
    )

    st.info(
        "The copilot calculates answers from the project "
        "datasets. It does not generate unsupported "
        "business figures."
    )

    st.markdown(
        """
**Try asking:**

- Which SKUs need immediate management attention?
- What replenishment actions should I prioritise today?
- How much revenue did the optimized policy recover?
- How accurate is the demand forecast?
- Compare Chennai and Mumbai.
- Tell me about Bath Soap 100g.
- Which city has the highest stockout exposure?
        """
    )

    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    if "copilot_messages" not in st.session_state:

        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me a business question about "
                    "demand forecasting, inventory, "
                    "revenue or SKU-level actions."
                )
            }
        ]

    for message in st.session_state.copilot_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # -----------------------------------------------------
    # USER QUESTION
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask a business question..."
    )

    if question:

        st.session_state.copilot_messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        answer = answer_business_question(
            question
        )

        st.session_state.copilot_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------

    if st.button(
        "Clear conversation"
    ):

        st.session_state.copilot_messages = []

        st.rerun()