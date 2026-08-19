from pathlib import Path

import py_compile
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "dashboard/app.py",
    "src/data/generate_data.py",
    "src/analysis/eda.py",
    "src/models/train_model.py",
    "src/inventory/recommend_inventory.py",
    "src/inventory/simulate_optimized_policy.py",
    "src/inventory/tune_inventory_policy.py",
    "data/processed/fmcg_sales.csv",
    "data/processed/demand_predictions.csv",
    "data/processed/inventory_holdout_results.csv",
    "data/processed/best_inventory_policy_simulation.csv"
]


PYTHON_FILES_TO_COMPILE = [
    "dashboard/app.py",
    "src/data/generate_data.py",
    "src/analysis/eda.py",
    "src/models/train_model.py",
    "src/inventory/recommend_inventory.py",
    "src/inventory/simulate_optimized_policy.py",
    "src/inventory/tune_inventory_policy.py"
]


def check_required_files():

    print("\n1. Checking required project files...")

    missing = []

    for relative_path in REQUIRED_FILES:

        file_path = (
            ROOT_DIR
            / relative_path
        )

        if not file_path.exists():
            missing.append(
                relative_path
            )

    if missing:

        print("\nMissing files:")

        for item in missing:
            print(f" - {item}")

        raise SystemExit(1)

    print("   PASS")


def check_python_syntax():

    print(
        "\n2. Checking Python syntax..."
    )

    for relative_path in (
        PYTHON_FILES_TO_COMPILE
    ):

        file_path = (
            ROOT_DIR
            / relative_path
        )

        py_compile.compile(
            str(file_path),
            doraise=True
        )

    print("   PASS")


def run_tests():

    print(
        "\n3. Running automated tests..."
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q"
        ],
        cwd=ROOT_DIR
    )

    if result.returncode != 0:

        print(
            "\nPROJECT HEALTH CHECK FAILED"
        )

        raise SystemExit(
            result.returncode
        )

    print("   PASS")


if __name__ == "__main__":

    print(
        "\nFMCG DECISION INTELLIGENCE"
    )

    print(
        "PROJECT HEALTH CHECK"
    )

    print("=" * 45)

    check_required_files()

    check_python_syntax()

    run_tests()

    print("\n" + "=" * 45)

    print(
        "ALL PROJECT HEALTH CHECKS PASSED"
    )

    print("=" * 45)