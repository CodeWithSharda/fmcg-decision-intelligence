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

# Track inventory separately for each store and product
inventory_levels = {}

for store in stores:
    for product in products:

        key = (store["store_id"], product["sku_id"])

        starting_inventory = int(
            base_demand[product["sku_id"]]
            * store["demand_factor"]
            * 14
        )

        inventory_levels[key] = starting_inventory
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

            # Identify this store-product inventory
            key = (store["store_id"], product["sku_id"])

            # Replenishment happens every Monday
            replenishment_qty = 0

            if date.dayofweek == 0:

                target_inventory = int(
                    base_demand[product["sku_id"]]
                    * store["demand_factor"]
                    * 7
                )

                replenishment_qty = max(
                    target_inventory - inventory_levels[key],
                    0
                )

                inventory_levels[key] += replenishment_qty

            # Inventory available at the beginning of the day
            opening_inventory = inventory_levels[key]

            # Customer demand for the day
            customer_demand = np.random.poisson(average_demand)

            # Actual sales cannot exceed available inventory
            units_sold = min(customer_demand, opening_inventory)

            # Demand we could not fulfil because stock was unavailable
            lost_sales = max(customer_demand - opening_inventory, 0)
            stockout_flag = lost_sales > 0

            # Remaining stock after today's sales
            ending_inventory = opening_inventory - units_sold

            # Update inventory for the next day
            inventory_levels[key] = ending_inventory

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
                "customer_demand": customer_demand,
                "opening_inventory": opening_inventory,
                "replenishment_qty": replenishment_qty,
                "units_sold": units_sold,
                "lost_sales": lost_sales,
                "stockout_flag": stockout_flag,
                "ending_inventory": ending_inventory,
                "revenue": revenue
            })

# Convert all sales records into a DataFrame
sales_df = pd.DataFrame(sales_records)

sales_df.to_csv("data/processed/fmcg_sales.csv", index=False)

print("\nSALES DATA PREVIEW")
print(sales_df.head(10))

print("\nTotal sales rows:", len(sales_df))

print("\nINVENTORY CHECK")
print("Total units replenished:", sales_df["replenishment_qty"].sum())
print("Total lost sales:", sales_df["lost_sales"].sum())
print("Rows with lost sales:", (sales_df["lost_sales"] > 0).sum())
print("Lowest ending inventory:", sales_df["ending_inventory"].min())