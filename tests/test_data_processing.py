from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def test_required_data_files_exist():
    required_files = [
        "areas.csv",
        "restaurants.csv",
        "dishes.csv",
        "raw_materials.csv",
        "suppliers.csv",
        "orders.csv",
        "order_items.csv",
        "inventory.csv",
    ]

    for filename in required_files:
        file_path = DATA_DIR / filename
        assert file_path.exists(), f"Missing file: {filename}"


def test_orders_data_is_valid():
    df = pd.read_csv(DATA_DIR / "orders.csv")

    required_columns = [
        "order_id",
        "order_datetime",
        "order_date",
        "customer_id",
        "restaurant_id",
        "area_id",
        "order_value",
        "discount",
        "payment_method",
        "order_status",
        "promotion",
    ]

    for column in required_columns:
        assert column in df.columns, f"Missing column: {column}"

    assert len(df) > 0
    assert df["order_id"].nunique() > 0


def test_order_items_data_is_valid():
    df = pd.read_csv(DATA_DIR / "order_items.csv")

    required_columns = [
        "order_item_id",
        "order_id",
        "dish_id",
        "quantity",
        "selling_price",
        "discount",
        "revenue",
        "ingredient_cost",
        "profit",
    ]

    for column in required_columns:
        assert column in df.columns, f"Missing column: {column}"

    assert len(df) > 0
    assert df["quantity"].min() >= 1
    assert df["revenue"].min() >= 0
    assert df["ingredient_cost"].min() >= 0


def test_business_calculations_are_valid():
    df = pd.read_csv(DATA_DIR / "order_items.csv")

    total_revenue = df["revenue"].sum()
    total_profit = df["profit"].sum()

    assert total_revenue > 0
    assert total_profit > 0
    assert total_profit <= total_revenue