from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

HOLDOUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "inventory_holdout_results.csv"
)

SIMULATION_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "best_inventory_policy_simulation.csv"
)

TUNING_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "inventory_policy_tuning.csv"
)


def load_holdout():
    return pd.read_csv(
        HOLDOUT_PATH
    )


def test_holdout_contains_both_policies():
    holdout = load_holdout()

    assert set(
        holdout["policy"]
    ) == {
        "Baseline",
        "Forecast-Driven"
    }


def test_forecast_policy_improves_service_level():
    holdout = load_holdout()

    baseline = (
        holdout[
            holdout["policy"]
            == "Baseline"
        ]
        .iloc[0]
    )

    optimized = (
        holdout[
            holdout["policy"]
            == "Forecast-Driven"
        ]
        .iloc[0]
    )

    assert (
        optimized["service_level"]
        > baseline["service_level"]
    )


def test_forecast_policy_reduces_lost_sales():
    holdout = load_holdout()

    baseline = (
        holdout[
            holdout["policy"]
            == "Baseline"
        ]
        .iloc[0]
    )

    optimized = (
        holdout[
            holdout["policy"]
            == "Forecast-Driven"
        ]
        .iloc[0]
    )

    assert (
        optimized["lost_sales"]
        < baseline["lost_sales"]
    )


def test_forecast_policy_reduces_stockout_events():
    holdout = load_holdout()

    baseline = (
        holdout[
            holdout["policy"]
            == "Baseline"
        ]
        .iloc[0]
    )

    optimized = (
        holdout[
            holdout["policy"]
            == "Forecast-Driven"
        ]
        .iloc[0]
    )

    assert (
        optimized["stockout_rows"]
        < baseline["stockout_rows"]
    )


def test_forecast_policy_reduces_average_inventory():
    holdout = load_holdout()

    baseline = (
        holdout[
            holdout["policy"]
            == "Baseline"
        ]
        .iloc[0]
    )

    optimized = (
        holdout[
            holdout["policy"]
            == "Forecast-Driven"
        ]
        .iloc[0]
    )

    assert (
        optimized["average_inventory"]
        < baseline["average_inventory"]
    )


def test_forecast_policy_increases_revenue():
    holdout = load_holdout()

    baseline = (
        holdout[
            holdout["policy"]
            == "Baseline"
        ]
        .iloc[0]
    )

    optimized = (
        holdout[
            holdout["policy"]
            == "Forecast-Driven"
        ]
        .iloc[0]
    )

    assert (
        optimized["revenue"]
        > baseline["revenue"]
    )


def test_simulation_inventory_never_negative():
    simulation = pd.read_csv(
        SIMULATION_PATH
    )

    assert (
        simulation[
            "ending_inventory"
        ] >= 0
    ).all()

    assert (
        simulation[
            "order_qty"
        ] >= 0
    ).all()

    assert (
        simulation[
            "lost_sales"
        ] >= 0
    ).all()


def test_simulation_demand_balance():
    simulation = pd.read_csv(
        SIMULATION_PATH
    )

    assert (
        simulation[
            "units_sold"
        ]
        <= simulation[
            "actual_demand"
        ]
    ).all()

    expected_lost_sales = (
        simulation[
            "actual_demand"
        ]
        - simulation[
            "units_sold"
        ]
    )

    assert np.allclose(
        simulation["lost_sales"],
        expected_lost_sales
    )


def test_policy_tuning_outputs_are_valid():
    tuning = pd.read_csv(
        TUNING_PATH
    )

    assert len(tuning) > 1

    assert (
        tuning[
            "service_level"
        ]
        .between(0, 100)
    ).all()

    assert (
        tuning[
            "average_inventory"
        ] >= 0
    ).all()

    assert (
        tuning[
            "lost_sales"
        ] >= 0
    ).all()