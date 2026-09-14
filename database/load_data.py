import sqlite3
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_FILE = PROJECT_ROOT / "restaurant.db"


# ============================================================
# CSV → SQLITE TABLE MAPPING
# ============================================================

FILES = {
    "areas.csv": "areas",
    "restaurants.csv": "restaurants",
    "dishes.csv": "dishes",
    "suppliers.csv": "suppliers",
    "raw_materials.csv": "raw_materials",
    "orders.csv": "orders",
    "order_items.csv": "order_items",
    "inventory.csv": "inventory",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_csv_data():

    print("=" * 60)
    print("LOADING CSV DATA INTO SQLITE")
    print("=" * 60)

    connection = sqlite3.connect(DATABASE_FILE)

    try:

        for filename, table_name in FILES.items():

            file_path = DATA_DIR / filename

            if not file_path.exists():
                print(f"ERROR: File not found - {file_path}")
                continue

            print(f"\nLoading {filename}...")

            df = pd.read_csv(file_path)

            # Replace existing data
            df.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False
            )

            print(
                f"  ✓ {table_name}: "
                f"{len(df):,} rows loaded"
            )

        connection.commit()

        print("\n" + "=" * 60)
        print("DATA LOADING COMPLETE")
        print("=" * 60)

    finally:
        connection.close()


# ============================================================
# VERIFY ROW COUNTS
# ============================================================

def verify_database():

    print("\n" + "=" * 60)
    print("DATABASE VERIFICATION")
    print("=" * 60)

    connection = sqlite3.connect(DATABASE_FILE)

    try:

        cursor = connection.cursor()

        tables = [
            "areas",
            "restaurants",
            "dishes",
            "suppliers",
            "raw_materials",
            "orders",
            "order_items",
            "inventory",
        ]

        total_rows = 0

        for table in tables:

            cursor.execute(
                f"SELECT COUNT(*) FROM {table}"
            )

            count = cursor.fetchone()[0]

            total_rows += count

            print(
                f"{table:<15} : {count:>10,} rows"
            )

        print("-" * 60)
        print(
            f"{'TOTAL':<15} : {total_rows:>10,} rows"
        )

    finally:
        connection.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    load_csv_data()
    verify_database()

    print("\n✓ SQLite database is ready for SQL analytics.")