import sqlite3
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = PROJECT_ROOT / "database" / "schema.sql"
DATABASE_FILE = PROJECT_ROOT / "restaurant.db"

# Read schema
with open(SCHEMA_FILE, "r", encoding="utf-8") as file:
    schema = file.read()

# Create database
connection = sqlite3.connect(DATABASE_FILE)

try:
    connection.executescript(schema)
    connection.commit()

    print("=" * 60)
    print("DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Database: {DATABASE_FILE}")
    print(f"Schema:   {SCHEMA_FILE}")

    # Verify tables
    cursor = connection.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """)

    tables = cursor.fetchall()

    print("\nTables created:")

    for table in tables:
        print(f"  ✓ {table[0]}")

    print(f"\nTotal tables: {len(tables)}")

finally:
    connection.close()

print("\nDatabase setup complete.")