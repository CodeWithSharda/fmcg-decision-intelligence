from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

SALES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "fmcg_sales.csv"
)


def load_sales():
    return pd.read_csv(
        SALES_PATH,
        parse_dates=["date"]
    )


def test_sales_dataset_exists_and_is_not_empty():
    assert SALES_PATH.exists()

    sales = load_sales()

    assert len(sales) > 0


def test_required_columns_exist():
    sales = load_sales()

    required_columns = {
        "date",
        "store_id",
        "city",
        "sku_id",
        "product_name",
        "category",
        "base_price",
        "promotion",
        "discount_pct",
        "selling_price",
        "customer_demand",
        "opening_inventory",
        "replenishment_qty",
        "units_sold",
        "lost_sales",
        "stockout_flag",
        "ending_inventory",
        "revenue"
    }

    assert required_columns.issubset(
        sales.columns
    )


def test_store_sku_date_is_unique():
    sales = load_sales()

    duplicates = sales.duplicated(
        subset=[
            "date",
            "store_id",
            "sku_id"
        ]
    )

    assert not duplicates.any()


def test_no_negative_business_values():
    sales = load_sales()

    non_negative_columns = [
        "base_price",
        "selling_price",
        "customer_demand",
        "opening_inventory",
        "replenishment_qty",
        "units_sold",
        "lost_sales",
        "ending_inventory",
        "revenue"
    ]

    for column in non_negative_columns:

        assert (
            sales[column] >= 0
        ).all(), (
            f"Negative values found in "
            f"{column}"
        )


def test_sales_cannot_exceed_customer_demand():
    sales = load_sales()

    assert (
        sales["units_sold"]
        <= sales["customer_demand"]
    ).all()


def test_lost_sales_equation():
    sales = load_sales()

    expected_lost_sales = (
        sales["customer_demand"]
        - sales["units_sold"]
    )

    assert np.allclose(
        sales["lost_sales"],
        expected_lost_sales
    )


def test_stockout_flag_matches_lost_sales():
    sales = load_sales()

    expected_stockout = (
        sales["lost_sales"] > 0
    )

    actual_stockout = sales[
        "stockout_flag"
    ]

    if actual_stockout.dtype != bool:

        actual_stockout = (
            actual_stockout
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False
                }
            )
        )

    assert actual_stockout.notna().all()

    assert np.array_equal(
        actual_stockout.to_numpy(),
        expected_stockout.to_numpy()
    )


def test_discounted_price_is_valid():
    sales = load_sales()

    assert (
        sales["selling_price"]
        <= sales["base_price"]
    ).all()

    assert (
        sales["discount_pct"]
        .between(0, 100)
    ).all()


def test_revenue_calculation():
    sales = load_sales()

    expected_revenue = (
        sales["units_sold"]
        * sales["selling_price"]
    )

    assert np.allclose(
        sales["revenue"],
        expected_revenue,
        atol=0.01
    )