import pandas as pd
from pathlib import Path


def load_data(data_dir="data"):
    """
    Load all restaurant datasets.
    """

    data_path = Path(data_dir)

    files = {
        "areas": "areas.csv",
        "restaurants": "restaurants.csv",
        "dishes": "dishes.csv",
        "raw_materials": "raw_materials.csv",
        "suppliers": "suppliers.csv",
        "orders": "orders.csv",
        "order_items": "order_items.csv",
        "inventory": "inventory.csv"
    }

    data = {}

    for name, filename in files.items():

        file_path = data_path / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        data[name] = pd.read_csv(file_path)

    return data


def prepare_transactions(data):
    """
    Create a clean transaction-level dataset.

    Financial values such as revenue, ingredient_cost
    and profit are taken from order_items.csv.
    """

    orders = data["orders"].copy()
    order_items = data["order_items"].copy()
    dishes = data["dishes"].copy()
    areas = data["areas"].copy()

    # --------------------------------------------------
    # Convert dates
    # --------------------------------------------------

    orders["order_datetime"] = pd.to_datetime(
        orders["order_datetime"],
        errors="coerce"
    )

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        errors="coerce"
    )

    # --------------------------------------------------
    # Merge order items with orders
    # --------------------------------------------------

    df = order_items.merge(
        orders,
        on="order_id",
        how="left",
        suffixes=("_item", "_order")
    )

    # --------------------------------------------------
    # Merge dish information
    # --------------------------------------------------

    df = df.merge(
        dishes,
        on="dish_id",
        how="left",
        suffixes=("", "_dish")
    )

    # --------------------------------------------------
    # Merge area information
    # --------------------------------------------------

    df = df.merge(
        areas,
        on="area_id",
        how="left"
    )

    # --------------------------------------------------
    # Convert important numeric columns
    # --------------------------------------------------

    numeric_columns = [
        "quantity",
        "selling_price",
        "discount_item",
        "revenue",
        "ingredient_cost",
        "profit",
        "order_value"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------
    # Fill missing values
    # --------------------------------------------------

    for column in [
        "quantity",
        "revenue",
        "ingredient_cost",
        "profit"
    ]:

        if column in df.columns:

            df[column] = df[column].fillna(0)

    # --------------------------------------------------
    # Profit margin
    # --------------------------------------------------

    df["profit_margin"] = (
        df["profit"] /
        df["revenue"].replace(0, pd.NA)
    )

    df["profit_margin"] = (
        df["profit_margin"]
        .fillna(0)
    )

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    df["year"] = df["order_date"].dt.year

    df["month"] = df["order_date"].dt.month

    df["month_name"] = (
        df["order_date"]
        .dt.month_name()
    )

    df["day_of_week"] = (
        df["order_date"].dt.dayofweek
    )

    df["day_name"] = (
        df["order_date"].dt.day_name()
    )

    df["week_of_year"] = (
        df["order_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # Create a clean discount column
    # --------------------------------------------------

    if "discount_item" in df.columns:

        df["discount"] = df["discount_item"]

    elif "discount" in df.columns:

        df["discount"] = df["discount"]

    else:

        df["discount"] = 0

    # --------------------------------------------------
    # Promotion
    # --------------------------------------------------

    if "promotion" not in df.columns:

        df["promotion"] = "None"

    df["promotion"] = (
        df["promotion"]
        .fillna("None")
        .astype(str)
    )

    # --------------------------------------------------
    # Restaurant / area information
    # --------------------------------------------------

    if "area_name" not in df.columns:

        df["area_name"] = (
            df["area_id"]
            .astype(str)
        )

    # --------------------------------------------------
    # Final cleaning
    # --------------------------------------------------

    df = df.sort_values(
        "order_datetime"
    ).reset_index(drop=True)

    return df


def print_summary(data, transactions):
    """
    Print project data summary.
    """

    print("\n========================================")
    print("       RESTAURANT DATA SUMMARY")
    print("========================================")

    for name, df in data.items():

        print(
            f"{name:<18} "
            f"{df.shape[0]:>10,} rows  "
            f"{df.shape[1]:>3} columns"
        )

    print("\n========================================")
    print("       TRANSACTION SUMMARY")
    print("========================================")

    print(
        "Transaction rows:",
        f"{len(transactions):,}"
    )

    print(
        "Unique orders:",
        f"{transactions['order_id'].nunique():,}"
    )

    print(
        "Unique dishes:",
        transactions["dish_id"].nunique()
    )

    print(
        "Unique areas:",
        transactions["area_id"].nunique()
    )

    print(
        "Total quantity sold:",
        f"{transactions['quantity'].sum():,.0f}"
    )

    print(
        "Total revenue:",
        f"₹{transactions['revenue'].sum():,.2f}"
    )

    print(
        "Total ingredient cost:",
        f"₹{transactions['ingredient_cost'].sum():,.2f}"
    )

    print(
        "Total profit:",
        f"₹{transactions['profit'].sum():,.2f}"
    )

    total_revenue = transactions["revenue"].sum()

    total_profit = transactions["profit"].sum()

    if total_revenue > 0:

        margin = (
            total_profit /
            total_revenue
        ) * 100

    else:

        margin = 0

    print(
        "Overall profit margin:",
        f"{margin:.2f}%"
    )

    print("\n========================================")
    print("       DATA PROCESSING COMPLETE")
    print("========================================")


if __name__ == "__main__":

    print("Loading restaurant datasets...")

    data = load_data("data")

    print("Preparing transaction dataset...")

    transactions = prepare_transactions(data)

    print_summary(
        data,
        transactions
    )

    print("\nFirst 5 transaction records:\n")

    print(
        transactions[
            [
                "order_id",
                "order_date",
                "dish_id",
                "quantity",
                "revenue",
                "ingredient_cost",
                "profit",
                "profit_margin",
                "area_id"
            ]
        ].head()
    )