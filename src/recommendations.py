import pandas as pd
import numpy as np

from data_processing import (
    load_data,
    prepare_transactions
)

from feature_engineering import (
    create_ml_dataset
)


DATA_DIR = "data"


def print_section(title):
    print("\n" + "=" * 60)
    print(f"       {title}")
    print("=" * 60)


def analyze_dish_performance(transactions, dishes):
    print_section("DISH PERFORMANCE ANALYSIS")

    dish_summary = (
        transactions
        .groupby("dish_id")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            avg_profit_margin=("profit_margin", "mean"),
            order_count=("order_id", "nunique")
        )
        .reset_index()
    )

    dish_summary = dish_summary.merge(
        dishes[
            [
                "dish_id",
                "dish_name",
                "category",
                "selling_price",
                "ingredient_cost"
            ]
        ],
        on="dish_id",
        how="left"
    )

    dish_summary["profit_per_unit"] = (
        dish_summary["total_profit"]
        / dish_summary["total_quantity"]
    )

    dish_summary = dish_summary.sort_values(
        "total_quantity",
        ascending=False
    )

    print("\nTOP 10 SELLING DISHES")
    print(
        dish_summary[
            [
                "dish_name",
                "category",
                "total_quantity",
                "total_revenue",
                "total_profit"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nLOWEST 5 SELLING DISHES")
    print(
        dish_summary[
            [
                "dish_name",
                "category",
                "total_quantity",
                "total_revenue",
                "total_profit"
            ]
        ]
        .tail(5)
        .sort_values("total_quantity")
        .to_string(index=False)
    )

    print("\nTOP 10 MOST PROFITABLE DISHES")
    print(
        dish_summary
        .sort_values("total_profit", ascending=False)
        [
            [
                "dish_name",
                "category",
                "total_quantity",
                "total_revenue",
                "total_profit",
                "avg_profit_margin"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return dish_summary


def analyze_sales_profitability(transactions):
    print_section("SALES VS PROFITABILITY")

    summary = (
        transactions
        .groupby("dish_id")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum")
        )
        .reset_index()
    )

    revenue_median = summary["total_revenue"].median()
    profit_median = summary["total_profit"].median()

    high_sales_low_profit = summary[
        (summary["total_revenue"] >= revenue_median)
        &
        (summary["total_profit"] < profit_median)
    ]

    print("\nHIGH-SALES / LOW-PROFIT DISHES")

    if high_sales_low_profit.empty:
        print("No high-sales / low-profit dishes identified.")
    else:
        print(
            high_sales_low_profit
            .sort_values("total_revenue", ascending=False)
            .to_string(index=False)
        )

    return high_sales_low_profit


def analyze_area_performance(transactions, areas):
    print_section("AREA PERFORMANCE")

    area_summary = (
        transactions
        .groupby("area_id")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            order_count=("order_id", "nunique")
        )
        .reset_index()
    )

    area_summary["profit_margin"] = (
        area_summary["total_profit"]
        / area_summary["total_revenue"]
    )

    # Only use columns that actually exist in areas.csv
    area_summary = area_summary.merge(
        areas[["area_id", "area_name", "city"]],
        on="area_id",
        how="left"
    )

    print("\nBEST PERFORMING AREAS")

    best_areas = (
        area_summary
        .sort_values("total_revenue", ascending=False)
        .head(5)
    )

    print(
        best_areas[
            [
                "area_name",
                "city",
                "total_revenue",
                "total_profit",
                "profit_margin",
                "order_count"
            ]
        ].to_string(index=False)
    )

    print("\nLOWEST PERFORMING AREAS")

    lowest_areas = (
        area_summary
        .sort_values("total_revenue", ascending=True)
        .head(5)
    )

    print(
        lowest_areas[
            [
                "area_name",
                "city",
                "total_revenue",
                "total_profit",
                "profit_margin",
                "order_count"
            ]
        ].to_string(index=False)
    )

    return area_summary

def analyze_inventory(inventory, raw_materials):
    print_section("INVENTORY INTELLIGENCE")

    inventory_summary = (
        inventory
        .groupby("material_id")
        .agg(
            total_purchased=("quantity_purchased", "sum"),
            total_used=("quantity_used", "sum"),
            total_wasted=("quantity_wasted", "sum"),
            avg_unit_cost=("unit_cost", "mean"),
            inventory_records=("inventory_id", "count")
        )
        .reset_index()
    )

    # Merge with raw material master data
    inventory_summary = inventory_summary.merge(
        raw_materials[
            [
                "material_id",
                "material_name",
                "category",
                "shelf_life_days",
                "reorder_level",
                "supplier_id"
            ]
        ],
        on="material_id",
        how="left"
    )

    # Calculate wastage rate
    inventory_summary["wastage_rate"] = np.where(
        inventory_summary["total_purchased"] > 0,
        (
            inventory_summary["total_wasted"]
            / inventory_summary["total_purchased"]
        ) * 100,
        0
    )

    # Calculate usage rate
    inventory_summary["usage_rate"] = np.where(
        inventory_summary["total_purchased"] > 0,
        (
            inventory_summary["total_used"]
            / inventory_summary["total_purchased"]
        ) * 100,
        0
    )

    # Estimate current stock
    inventory_summary["estimated_stock"] = (
        inventory_summary["total_purchased"]
        - inventory_summary["total_used"]
        - inventory_summary["total_wasted"]
    )

    print("\nINVENTORY SUMMARY")

    print(
        inventory_summary[
            [
                "material_name",
                "category",
                "total_purchased",
                "total_used",
                "total_wasted",
                "estimated_stock"
            ]
        ]
        .sort_values("total_used", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTOP MATERIALS BY WASTAGE")

    print(
        inventory_summary
        .sort_values("total_wasted", ascending=False)
        [
            [
                "material_name",
                "category",
                "total_purchased",
                "total_wasted",
                "wastage_rate"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nHIGHEST WASTAGE RATE MATERIALS")

    print(
        inventory_summary
        .sort_values("wastage_rate", ascending=False)
        [
            [
                "material_name",
                "category",
                "wastage_rate",
                "total_wasted"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nLOW STOCK MATERIALS")

    low_stock = inventory_summary[
        inventory_summary["estimated_stock"]
        <= inventory_summary["reorder_level"]
    ]

    if low_stock.empty:
        print("No materials currently below reorder level.")
    else:
        print(
            low_stock[
                [
                    "material_name",
                    "estimated_stock",
                    "reorder_level",
                    "total_used"
                ]
            ]
            .sort_values("estimated_stock")
            .head(10)
            .to_string(index=False)
        )

    return inventory_summary


def analyze_demand_trend(ml_data):
    print_section("DEMAND TREND ANALYSIS")

    demand_summary = (
        ml_data
        .groupby("dish_id")
        .agg(
            avg_demand=("demand", "mean"),
            total_demand=("demand", "sum"),
            avg_growth=("demand_growth", "mean"),
            latest_demand=("demand", "last")
        )
        .reset_index()
    )

    print("\nFASTEST GROWING DISHES")

    print(
        demand_summary
        .sort_values("avg_growth", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nLOWEST GROWTH DISHES")

    print(
        demand_summary
        .sort_values("avg_growth")
        .head(5)
        .to_string(index=False)
    )

    return demand_summary


def generate_recommendations(
    dish_summary,
    area_summary,
    inventory_summary,
    high_sales_low_profit
):
    print_section("BUSINESS RECOMMENDATIONS")

    recommendations = []

    # Recommendation 1
    top_dish = dish_summary.sort_values(
        "total_quantity",
        ascending=False
    ).iloc[0]

    recommendations.append(
        f"1. Increase inventory planning for "
        f"{top_dish['dish_name']} because it is the highest-volume dish."
    )

    # Recommendation 2
    profitable_dish = dish_summary.sort_values(
        "total_profit",
        ascending=False
    ).iloc[0]

    recommendations.append(
        f"2. Promote {profitable_dish['dish_name']} because it "
        f"generates the highest total profit."
    )

    # Recommendation 3
    if not high_sales_low_profit.empty:
        dish_id = high_sales_low_profit.iloc[0]["dish_id"]

        matching = dish_summary[
            dish_summary["dish_id"] == dish_id
        ]

        if not matching.empty:
            dish_name = matching.iloc[0]["dish_name"]

            recommendations.append(
                f"3. Review pricing, discounts or ingredient costs "
                f"for {dish_name} because it has high sales but "
                f"relatively low profitability."
            )

    # Recommendation 4
    high_wastage = inventory_summary.sort_values(
        "wastage_rate",
        ascending=False
    ).iloc[0]

    recommendations.append(
        f"4. Investigate wastage of "
        f"{high_wastage['material_name']} because it has "
        f"one of the highest wastage rates."
    )

    # Recommendation 5
    best_area = area_summary.sort_values(
        "total_revenue",
        ascending=False
    ).iloc[0]

    recommendations.append(
        f"5. Prioritize marketing and expansion opportunities "
        f"in {best_area['area_name']} because it generates the "
        f"highest revenue."
    )

    for recommendation in recommendations:
        print("\n" + recommendation)

    return recommendations


def main():

    print_section("RESTAURANT BUSINESS INTELLIGENCE")

    print("\n1. Loading datasets...")

    data = load_data(DATA_DIR)

    print("Datasets loaded successfully.")

    print("\n2. Preparing transaction data...")

    transactions = prepare_transactions(data)

    print(
        f"Transaction rows: {len(transactions):,}"
    )

    print("\n3. Creating ML dataset...")

    ml_data = create_ml_dataset(transactions)

    print(
        f"ML rows: {len(ml_data):,}"
    )

    print("\n4. Running business analysis...")

    dish_summary = analyze_dish_performance(
        transactions,
        data["dishes"]
    )

    high_sales_low_profit = analyze_sales_profitability(
        transactions
    )

    area_summary = analyze_area_performance(
        transactions,
        data["areas"]
    )

    inventory_summary = analyze_inventory(
        data["inventory"],
        data["raw_materials"]
    )

    demand_summary = analyze_demand_trend(
        ml_data
    )

    recommendations = generate_recommendations(
        dish_summary,
        area_summary,
        inventory_summary,
        high_sales_low_profit
    )

    print_section("ANALYSIS COMPLETE")

    print("\nBusiness intelligence analysis completed successfully.")

    print("\nGenerated analysis sections:")
    print("1. Dish Performance")
    print("2. Sales vs Profitability")
    print("3. Area Performance")
    print("4. Inventory Intelligence")
    print("5. Demand Trends")
    print("6. Business Recommendations")


if __name__ == "__main__":
    main()