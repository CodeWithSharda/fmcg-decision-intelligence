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

PREDICTIONS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "demand_predictions.csv"
)


def load_predictions():
    return pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["date"]
    )


def test_prediction_file_exists():
    assert PREDICTIONS_PATH.exists()


def test_predictions_are_complete():
    predictions = load_predictions()

    required = {
        "date",
        "store_id",
        "sku_id",
        "customer_demand",
        "predicted_demand",
        "absolute_error"
    }

    assert required.issubset(
        predictions.columns
    )

    assert len(predictions) > 0

    assert not predictions[
        list(required)
    ].isnull().any().any()


def test_predictions_are_non_negative():
    predictions = load_predictions()

    assert (
        predictions[
            "predicted_demand"
        ] >= 0
    ).all()


def test_absolute_error_is_correct():
    predictions = load_predictions()

    expected_error = np.abs(
        predictions["customer_demand"]
        - predictions["predicted_demand"]
    )

    assert np.allclose(
        predictions["absolute_error"],
        expected_error,
        atol=0.01
    )


def test_xgboost_beats_lag7_baseline():
    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"]
    )

    predictions = load_predictions()

    sales = sales.sort_values(
        [
            "store_id",
            "sku_id",
            "date"
        ]
    )

    sales["lag_7"] = (
        sales
        .groupby(
            [
                "store_id",
                "sku_id"
            ]
        )["customer_demand"]
        .shift(7)
    )

    comparison = predictions.merge(
        sales[
            [
                "date",
                "store_id",
                "sku_id",
                "lag_7"
            ]
        ],
        on=[
            "date",
            "store_id",
            "sku_id"
        ],
        how="left"
    )

    comparison = (
        comparison
        .dropna(
            subset=["lag_7"]
        )
    )

    model_mae = np.mean(
        np.abs(
            comparison[
                "customer_demand"
            ]
            - comparison[
                "predicted_demand"
            ]
        )
    )

    baseline_mae = np.mean(
        np.abs(
            comparison[
                "customer_demand"
            ]
            - comparison[
                "lag_7"
            ]
        )
    )

    assert model_mae < baseline_mae


def test_forecast_mape_is_reasonable():
    predictions = load_predictions()

    nonzero = (
        predictions[
            "customer_demand"
        ] > 0
    )

    mape = np.mean(
        np.abs(
            (
                predictions.loc[
                    nonzero,
                    "customer_demand"
                ]
                - predictions.loc[
                    nonzero,
                    "predicted_demand"
                ]
            )
            /
            predictions.loc[
                nonzero,
                "customer_demand"
            ]
        )
    ) * 100

    # Portfolio-quality guardrail.
    # If future data/model changes push MAPE
    # above this threshold, investigate.
    assert mape < 25