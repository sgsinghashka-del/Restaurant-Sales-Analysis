import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_FILE = PROJECT_ROOT / "restaurant.db"


def run_query(connection, title, query):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    cursor = connection.cursor()
    cursor.execute(query)

    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()

    print(" | ".join(columns))
    print("-" * 70)

    for row in rows[:10]:
        print(" | ".join(str(value) for value in row))

    print(f"\nRows returned: {len(rows)}")


connection = sqlite3.connect(DATABASE_FILE)

try:

    run_query(
        connection,
        "OVERALL BUSINESS PERFORMANCE",
        """
        SELECT
            COUNT(DISTINCT order_id) AS total_orders,
            SUM(quantity) AS total_items_sold,
            ROUND(SUM(revenue), 2) AS total_revenue,
            ROUND(SUM(profit), 2) AS total_profit,
            ROUND(
                SUM(profit) * 100.0 /
                NULLIF(SUM(revenue), 0),
                2
            ) AS profit_margin_percentage
        FROM order_items;
        """
    )

    run_query(
        connection,
        "TOP 10 SELLING DISHES",
        """
        SELECT
            d.dish_name,
            d.category,
            SUM(oi.quantity) AS quantity_sold,
            ROUND(SUM(oi.revenue), 2) AS revenue,
            ROUND(SUM(oi.profit), 2) AS profit
        FROM order_items oi
        JOIN dishes d
            ON oi.dish_id = d.dish_id
        GROUP BY d.dish_id, d.dish_name, d.category
        ORDER BY quantity_sold DESC
        LIMIT 10;
        """
    )

    run_query(
        connection,
        "CATEGORY-WISE SALES",
        """
        SELECT
            d.category,
            SUM(oi.quantity) AS quantity_sold,
            ROUND(SUM(oi.revenue), 2) AS revenue,
            ROUND(SUM(oi.profit), 2) AS profit
        FROM order_items oi
        JOIN dishes d
            ON oi.dish_id = d.dish_id
        GROUP BY d.category
        ORDER BY revenue DESC;
        """
    )

    run_query(
        connection,
        "AREA-WISE PERFORMANCE",
        """
        SELECT
            a.area_name,
            a.city,
            COUNT(DISTINCT o.order_id) AS orders,
            ROUND(SUM(oi.revenue), 2) AS revenue,
            ROUND(SUM(oi.profit), 2) AS profit
        FROM order_items oi
        JOIN orders o
            ON oi.order_id = o.order_id
        JOIN areas a
            ON o.area_id = a.area_id
        GROUP BY a.area_id, a.area_name, a.city
        ORDER BY revenue DESC;
        """
    )

finally:
    connection.close()

print("\n" + "=" * 70)
print("SQL ANALYTICS TEST COMPLETE")
print("=" * 70)