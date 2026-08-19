import pandas as pd
import numpy as np

products = [
    {
        "sku_id": "SKU001",
        "product_name": "Shampoo 340ml",
        "category": "Personal Care",
        "base_price": 329
    },
    {
        "sku_id": "SKU002",
        "product_name": "Bath Soap 100g",
        "category": "Personal Care",
        "base_price": 55
    },
    {
        "sku_id": "SKU003",
        "product_name": "Laundry Detergent 1kg",
        "category": "Home Care",
        "base_price": 220
    },
    {
        "sku_id": "SKU004",
        "product_name": "Dishwash Liquid 500ml",
        "category": "Home Care",
        "base_price": 125
    },
    {
        "sku_id": "SKU005",
        "product_name": "Instant Coffee 100g",
        "category": "Beverages",
        "base_price": 185
    }
]

products_df = pd.DataFrame(products)

print(products_df)

stores = [
    {
        "store_id": "ST001",
        "city": "Chennai",
        "store_size": "Large",
        "demand_factor": 1.20
    },
    {
        "store_id": "ST002",
        "city": "Bengaluru",
        "store_size": "Large",
        "demand_factor": 1.25
    },
    {
        "store_id": "ST003",
        "city": "Mumbai",
        "store_size": "Large",
        "demand_factor": 1.30
    },
    {
        "store_id": "ST004",
        "city": "Delhi",
        "store_size": "Medium",
        "demand_factor": 1.10
    },
    {
        "store_id": "ST005",
        "city": "Hyderabad",
        "store_size": "Medium",
        "demand_factor": 1.05
    }
]

stores_df = pd.DataFrame(stores)

print("\nSTORE MASTER")
print(stores_df)

# Create daily dates from January 2024 to July 2026
dates = pd.date_range(
    start="2024-01-01",
    end="2026-07-31",
    freq="D"
)

print("\nDATE RANGE")
print("First date:", dates[0])
print("Last date:", dates[-1])
print("Total days:", len(dates))

# Make random sales numbers reproducible
np.random.seed(42)

# Normal daily demand for each product
base_demand = {
    "SKU001": 20,
    "SKU002": 35,
    "SKU003": 28,
    "SKU004": 24,
    "SKU005": 18
}

# Empty list where all sales records will be stored
sales_records = []

# Go through every date
for date in dates:

    # Go through every store
    for store in stores:

        # Go through every product
        for product in products:

             # Saturday = 5 and Sunday = 6
            is_weekend = date.dayofweek >= 5

            # Demand is 15% higher on weekends
            weekend_factor = 1.15 if is_weekend else 1.00

            average_demand = (
                base_demand[product["sku_id"]]
                * store["demand_factor"]
                * weekend_factor
            )

            # Around 10% of product-store-days have a promotion
            is_promotion = np.random.random() < 0.10

            discount_pct = np.random.choice([5, 10, 15, 20]) if is_promotion else 0

            selling_price = round(
            product["base_price"] * (1 - discount_pct / 100),
            2
            )
            # Bigger discounts create a larger increase in demand
            discount_factor = 1 + (discount_pct / 100) * 1.5

            average_demand = average_demand * discount_factor

            units_sold = np.random.poisson(average_demand)
            revenue = round(units_sold * selling_price, 2)
            sales_records.append({
                "date": date,
                "day_of_week": date.day_name(),
                "is_weekend": is_weekend,
                "store_id": store["store_id"],
                "city": store["city"],
                "sku_id": product["sku_id"],
                "product_name": product["product_name"],
                "category": product["category"],
                "base_price": product["base_price"],
                "promotion": is_promotion,
                "discount_pct": discount_pct,
                "selling_price": selling_price,
                "units_sold": units_sold,
                "revenue": revenue
            })

# Convert all sales records into a DataFrame
sales_df = pd.DataFrame(sales_records)

sales_df.to_csv("data/processed/fmcg_sales.csv", index=False)

print("\nSALES DATA PREVIEW")
print(sales_df.head(10))

print("\nTotal sales rows:", len(sales_df))