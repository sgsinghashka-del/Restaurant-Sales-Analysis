import pandas as pd
import numpy as np

from src.data_processing import (
    load_data,
    prepare_transactions
)


def create_daily_demand(transactions):
    """
    Convert transaction-level data into daily
    dish-level demand data.

    One row represents:

        Date + Area + Dish

    Target variable:

        demand = total quantity sold
    """

    df = transactions.copy()

    # --------------------------------------------------
    # Aggregate daily demand
    # --------------------------------------------------

    daily = (
        df.groupby(
            [
                "order_date",
                "area_id",
                "dish_id"
            ]
        )
        .agg(
            demand=("quantity", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            ingredient_cost=("ingredient_cost", "sum"),
            avg_discount=("discount", "mean"),
            promotion_count=("promotion", "nunique"),
            order_count=("order_id", "nunique")
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------

    daily = daily.sort_values(
        [
            "area_id",
            "dish_id",
            "order_date"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    daily["year"] = (
        daily["order_date"].dt.year
    )

    daily["month"] = (
        daily["order_date"].dt.month
    )

    daily["day_of_week"] = (
        daily["order_date"].dt.dayofweek
    )

    daily["day_of_month"] = (
        daily["order_date"].dt.day
    )

    daily["week_of_year"] = (
        daily["order_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    daily["is_weekend"] = (
        daily["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # Time-based demand features
    # --------------------------------------------------

    group_columns = [
        "area_id",
        "dish_id"
    ]

    # Previous day demand
    daily["lag_1"] = (
        daily
        .groupby(group_columns)["demand"]
        .shift(1)
    )

    # Previous week demand
    daily["lag_7"] = (
        daily
        .groupby(group_columns)["demand"]
        .shift(7)
    )

    # Previous 14-day demand
    daily["lag_14"] = (
        daily
        .groupby(group_columns)["demand"]
        .shift(14)
    )

    # Previous 30-day demand
    daily["lag_30"] = (
        daily
        .groupby(group_columns)["demand"]
        .shift(30)
    )

    # --------------------------------------------------
    # Rolling demand features
    # --------------------------------------------------

    daily["rolling_7"] = (
        daily
        .groupby(group_columns)["demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=7,
                min_periods=3
            )
            .mean()
        )
    )

    daily["rolling_14"] = (
        daily
        .groupby(group_columns)["demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=14,
                min_periods=5
            )
            .mean()
        )
    )

    daily["rolling_30"] = (
        daily
        .groupby(group_columns)["demand"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=30,
                min_periods=7
            )
            .mean()
        )
    )

    # --------------------------------------------------
    # Rolling revenue
    # --------------------------------------------------

    daily["revenue_rolling_7"] = (
        daily
        .groupby(group_columns)["revenue"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=7,
                min_periods=3
            )
            .mean()
        )
    )

    # --------------------------------------------------
    # Rolling profit
    # --------------------------------------------------

    daily["profit_rolling_7"] = (
        daily
        .groupby(group_columns)["profit"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=7,
                min_periods=3
            )
            .mean()
        )
    )

    # --------------------------------------------------
    # Demand trend
    # --------------------------------------------------

    daily["demand_trend"] = (
        daily["rolling_7"] -
        daily["rolling_30"]
    )

    # --------------------------------------------------
    # Demand growth
    # --------------------------------------------------

    daily["demand_growth"] = (
        (
            daily["lag_7"] -
            daily["lag_30"]
        )
        /
        daily["lag_30"].replace(0, np.nan)
    )

    # --------------------------------------------------
    # Profit margin
    # --------------------------------------------------

    daily["profit_margin"] = (
        daily["profit"] /
        daily["revenue"].replace(0, np.nan)
    )

    # --------------------------------------------------
    # Clean infinite values
    # --------------------------------------------------

    daily = daily.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------
    # Remove rows where historical features
    # are unavailable
    # --------------------------------------------------

    required_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_7",
        "rolling_14",
        "rolling_30"
    ]

    daily = daily.dropna(
        subset=required_features
    )

    # --------------------------------------------------
    # Reset index
    # --------------------------------------------------

    daily = daily.reset_index(
        drop=True
    )

    return daily


def get_feature_columns():
    """
    Return ML feature columns.

    Target:
        demand
    """

    features = [
        "area_id",
        "dish_id",

        # Calendar
        "year",
        "month",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_weekend",

        # Pricing / promotion
        "avg_discount",
        "promotion_count",

        # Historical demand
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",

        # Rolling demand
        "rolling_7",
        "rolling_14",
        "rolling_30",

        # Business trends
        "revenue_rolling_7",
        "profit_rolling_7",
        "demand_trend",
        "demand_growth"
    ]

    return features


def create_ml_dataset(transactions):
    """
    Create final dataset for machine learning.
    """

    daily = create_daily_demand(
        transactions
    )

    features = get_feature_columns()

    target = "demand"

    # Make sure all feature columns exist
    missing_columns = [
        column
        for column in features
        if column not in daily.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing feature columns: "
            + str(missing_columns)
        )

    ml_data = daily[
        [
            "order_date",
            "area_id",
            "dish_id",
            target
        ]
        + features
    ].copy()

    # Remove duplicate columns
    ml_data = ml_data.loc[
        :,
        ~ml_data.columns.duplicated()
    ]

    return ml_data


if __name__ == "__main__":

    print("\n========================================")
    print("       FEATURE ENGINEERING")
    print("========================================")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\n1. Loading datasets...")

    data = load_data("data")

    print("Datasets loaded successfully.")

    # --------------------------------------------------
    # Prepare transactions
    # --------------------------------------------------

    print("\n2. Preparing transactions...")

    transactions = prepare_transactions(
        data
    )

    print(
        "Transaction rows:",
        f"{len(transactions):,}"
    )

    # --------------------------------------------------
    # Create ML dataset
    # --------------------------------------------------

    print("\n3. Creating ML features...")

    ml_data = create_ml_dataset(
        transactions
    )

    # --------------------------------------------------
    # Display information
    # --------------------------------------------------

    print("\n========================================")
    print("       ML DATASET SUMMARY")
    print("========================================")

    print(
        "Rows:",
        f"{len(ml_data):,}"
    )

    print(
        "Columns:",
        len(ml_data.columns)
    )

    print(
        "Features:",
        len(get_feature_columns())
    )

    print(
        "Target:",
        "demand"
    )

    # --------------------------------------------------
    # Date range
    # --------------------------------------------------

    print(
        "\nDate range:"
    )

    print(
        "Start:",
        ml_data["order_date"].min()
    )

    print(
        "End:",
        ml_data["order_date"].max()
    )

    # --------------------------------------------------
    # Demand statistics
    # --------------------------------------------------

    print(
        "\nDemand statistics:"
    )

    print(
        ml_data["demand"].describe()
    )

    # --------------------------------------------------
    # Feature list
    # --------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "       FEATURE LIST"
    )

    print(
        "========================================"
    )

    for i, feature in enumerate(
        get_feature_columns(),
        start=1
    ):

        print(
            f"{i:02d}. {feature}"
        )

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "       DATA PREVIEW"
    )

    print(
        "========================================"
    )

    print(
        ml_data.head()
    )

    print(
        "\nFeature engineering completed successfully."
    )