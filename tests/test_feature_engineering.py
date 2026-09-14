from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_processing import load_data, prepare_transactions
from src.feature_engineering import create_ml_dataset


def get_ml_dataset():
    """
    Build the ML dataset using the same preprocessing
    pipeline used by the application.
    """

    data = load_data()

    transactions = prepare_transactions(data)

    assert isinstance(
        transactions,
        pd.DataFrame
    ), "prepare_transactions() must return a DataFrame"

    assert len(transactions) > 0, (
        "prepare_transactions() returned an empty DataFrame"
    )

    transactions["order_date"] = pd.to_datetime(
        transactions["order_date"],
        errors="coerce"
    )

    assert transactions["order_date"].notna().all(), (
        "order_date contains invalid or missing dates"
    )

    ml_data = create_ml_dataset(transactions)

    return ml_data


def test_feature_dataset_is_created():

    ml_data = get_ml_dataset()

    assert ml_data is not None
    assert isinstance(ml_data, pd.DataFrame)
    assert len(ml_data) > 0


def test_demand_target_exists():

    ml_data = get_ml_dataset()

    assert "demand" in ml_data.columns

    assert ml_data["demand"].notna().all()

    assert ml_data["demand"].min() >= 0

    assert ml_data["demand"].sum() > 0


def test_required_features_exist():

    ml_data = get_ml_dataset()

    required_features = [
        "area_id",
        "dish_id",
        "year",
        "month",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_weekend",
        "avg_discount",
        "promotion_count",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_7",
        "rolling_14",
        "rolling_30",
        "revenue_rolling_7",
        "profit_rolling_7",
        "demand_trend",
        "demand_growth",
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in ml_data.columns
    ]

    assert not missing_features, (
        f"Missing features: {missing_features}"
    )


def test_feature_dataset_has_expected_size():

    ml_data = get_ml_dataset()

    assert len(ml_data) > 100000

    assert ml_data["demand"].sum() > 0

    assert ml_data["area_id"].nunique() > 1

    assert ml_data["dish_id"].nunique() > 1