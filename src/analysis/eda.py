from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

DATA_PATH = Path("data/processed/fmcg_sales.csv")
FIGURE_PATH = Path("reports/figures")

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

sales_df = pd.read_csv(DATA_PATH, parse_dates=["date"])


# ---------------------------------------------------------
# BASIC DATASET INFORMATION
# ---------------------------------------------------------

print("\nFMCG DECISION INTELLIGENCE - EDA")
print("=" * 45)

print("\nDATASET OVERVIEW")
print("Rows:", len(sales_df))
print("Columns:", len(sales_df.columns))
print("Start date:", sales_df["date"].min().date())
print("End date:", sales_df["date"].max().date())

print("\nMissing values:")
print(sales_df.isnull().sum())


# ---------------------------------------------------------
# BUSINESS KPIs
# ---------------------------------------------------------

total_revenue = sales_df["revenue"].sum()
total_units_sold = sales_df["units_sold"].sum()
total_customer_demand = sales_df["customer_demand"].sum()
total_lost_sales = sales_df["lost_sales"].sum()

stockout_rate = sales_df["stockout_flag"].mean() * 100

demand_fulfilment_rate = (
    total_units_sold / total_customer_demand
) * 100


print("\nBUSINESS KPIs")
print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Total Units Sold: {total_units_sold:,}")
print(f"Total Customer Demand: {total_customer_demand:,}")
print(f"Total Lost Sales: {total_lost_sales:,}")
print(f"Stockout Rate: {stockout_rate:.2f}%")
print(f"Demand Fulfilment Rate: {demand_fulfilment_rate:.2f}%")


# ---------------------------------------------------------
# 1. REVENUE BY CITY
# ---------------------------------------------------------

revenue_by_city = (
    sales_df.groupby("city")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nREVENUE BY CITY")
print(revenue_by_city)

plt.figure(figsize=(9, 5))
revenue_by_city.plot(kind="bar")
plt.title("Revenue by City")
plt.xlabel("City")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURE_PATH / "revenue_by_city.png")
plt.close()


# ---------------------------------------------------------
# 2. UNITS SOLD BY PRODUCT
# ---------------------------------------------------------

sales_by_product = (
    sales_df.groupby("product_name")["units_sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\nUNITS SOLD BY PRODUCT")
print(sales_by_product)

plt.figure(figsize=(10, 5))
sales_by_product.plot(kind="bar")
plt.title("Units Sold by Product")
plt.xlabel("Product")
plt.ylabel("Units Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "units_sold_by_product.png")
plt.close()


# ---------------------------------------------------------
# 3. PROMOTION IMPACT
# ---------------------------------------------------------

promotion_impact = (
    sales_df.groupby("promotion")["customer_demand"]
    .mean()
)

print("\nAVERAGE CUSTOMER DEMAND - PROMOTION IMPACT")
print(promotion_impact)

plt.figure(figsize=(7, 5))
promotion_impact.plot(kind="bar")
plt.title("Average Demand: Promotion vs No Promotion")
plt.xlabel("Promotion Active")
plt.ylabel("Average Customer Demand")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_PATH / "promotion_impact.png")
plt.close()


# ---------------------------------------------------------
# 4. WEEKEND IMPACT
# ---------------------------------------------------------

weekend_impact = (
    sales_df.groupby("is_weekend")["customer_demand"]
    .mean()
)

print("\nAVERAGE CUSTOMER DEMAND - WEEKEND IMPACT")
print(weekend_impact)

plt.figure(figsize=(7, 5))
weekend_impact.plot(kind="bar")
plt.title("Average Demand: Weekday vs Weekend")
plt.xlabel("Weekend")
plt.ylabel("Average Customer Demand")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_PATH / "weekend_impact.png")
plt.close()


# ---------------------------------------------------------
# 5. STOCKOUT RATE BY PRODUCT
# ---------------------------------------------------------

stockout_by_product = (
    sales_df.groupby("product_name")["stockout_flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nSTOCKOUT RATE BY PRODUCT (%)")
print(stockout_by_product)

plt.figure(figsize=(10, 5))
stockout_by_product.plot(kind="bar")
plt.title("Stockout Rate by Product")
plt.xlabel("Product")
plt.ylabel("Stockout Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "stockout_rate_by_product.png")
plt.close()


# ---------------------------------------------------------
# 6. MONTHLY REVENUE TREND
# ---------------------------------------------------------

monthly_revenue = (
    sales_df.set_index("date")
    .resample("ME")["revenue"]
    .sum()
)

plt.figure(figsize=(11, 5))
monthly_revenue.plot()
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "monthly_revenue_trend.png")
plt.close()


print("\nEDA COMPLETE")
print("Charts saved in reports/figures/")