import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import (
    load_data,
    prepare_transactions
)

from src.feature_engineering import (
    create_ml_dataset
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Restaurant Sales Intelligence",
    page_icon="🍽️",
    layout="wide"
)


DATA_DIR = "data"
MODEL_DIR = Path("models")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_restaurant_data():

    data = load_data(DATA_DIR)

    transactions = prepare_transactions(data)

    ml_data = create_ml_dataset(transactions)

    return data, transactions, ml_data


@st.cache_resource
def load_model():

    model_path = MODEL_DIR / "xgboost_demand_model.pkl"

    if not model_path.exists():
        return None

    return joblib.load(model_path)


@st.cache_data
def load_model_metadata():

    metrics_path = MODEL_DIR / "metrics.json"
    encoder_path = MODEL_DIR / "encoders.json"

    metrics = {}
    encoders = {}

    if metrics_path.exists():

        with open(metrics_path, "r") as file:
            metrics = json.load(file)

    if encoder_path.exists():

        with open(encoder_path, "r") as file:
            encoders = json.load(file)

    return metrics, encoders


# ============================================================
# LOAD EVERYTHING
# ============================================================

data, transactions, ml_data = load_restaurant_data()

model = load_model()

metrics, encoders = load_model_metadata()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🍽️ Restaurant Intelligence")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Dashboard",
        "Sales Analysis",
        "Profitability",
        "Dish Performance",
        "Area Performance",
        "Inventory Intelligence",
        "Demand Forecast",
        "ML Performance",
        "Business Recommendations"
    ]
)


# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

transactions["order_date"] = pd.to_datetime(
    transactions["order_date"]
)

min_date = transactions["order_date"].min()
max_date = transactions["order_date"].max()

date_range = st.sidebar.date_input(
    "Order Date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    filtered_transactions = transactions[
        (transactions["order_date"] >= start_date)
        &
        (transactions["order_date"] <= end_date)
    ].copy()

else:

    filtered_transactions = transactions.copy()


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.title("🍽️ Restaurant Sales Intelligence Platform")

    st.markdown(
        "### Executive Overview"
    )

    st.caption(
        "Sales, profitability, demand and inventory intelligence"
    )

    total_revenue = filtered_transactions["revenue"].sum()

    total_profit = filtered_transactions["profit"].sum()

    total_orders = filtered_transactions["order_id"].nunique()

    total_quantity = filtered_transactions["quantity"].sum()

    profit_margin = (
        total_profit / total_revenue
        if total_revenue > 0
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Revenue",
        f"₹{total_revenue:,.0f}"
    )

    col2.metric(
        "Total Profit",
        f"₹{total_profit:,.0f}"
    )

    col3.metric(
        "Orders",
        f"{total_orders:,}"
    )

    col4.metric(
        "Units Sold",
        f"{total_quantity:,}"
    )

    col5.metric(
        "Profit Margin",
        f"{profit_margin * 100:.2f}%"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    daily_sales = (
        filtered_transactions
        .groupby("order_date")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum")
        )
        .reset_index()
    )

    with col1:

        fig = px.line(
            daily_sales,
            x="order_date",
            y="revenue",
            title="Daily Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.line(
            daily_sales,
            x="order_date",
            y="profit",
            title="Daily Profit"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Top Selling Dishes")

    top_dishes = (
        filtered_transactions
        .groupby("dish_id")
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum")
        )
        .reset_index()
        .merge(
            data["dishes"][
                ["dish_id", "dish_name", "category"]
            ],
            on="dish_id",
            how="left"
        )
        .sort_values(
            "quantity",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_dishes,
        use_container_width=True
    )


# ============================================================
# SALES ANALYSIS
# ============================================================

elif page == "Sales Analysis":

    st.title("Sales Analysis")
    st.markdown("Analyze restaurant sales by category, dish, area and payment method.")

    # ---------------------------------------------------------
    # FILTERED TRANSACTIONS
    # ---------------------------------------------------------

    sales_df = data["order_items"].copy()

    sales_df["order_id"] = sales_df["order_id"].astype(str)

    orders_sales = data["orders"].copy()
    orders_sales["order_id"] = orders_sales["order_id"].astype(str)

    sales_df = sales_df.merge(
        orders_sales[
            [
                "order_id",
                "order_date",
                "restaurant_id",
                "area_id",
                "payment_method"
            ]
        ],
        on="order_id",
        how="left"
    )

    sales_df["order_date"] = pd.to_datetime(
        sales_df["order_date"],
        errors="coerce"
    )

    # Apply dashboard date filter
    sales_df = sales_df[
        (sales_df["order_date"] >= start_date)
        & (sales_df["order_date"] <= end_date)
    ].copy()

    # ---------------------------------------------------------
    # ADD DISH INFORMATION
    # ---------------------------------------------------------

    dishes_sales = data["dishes"][
        [
            "dish_id",
            "dish_name",
            "category"
        ]
    ].copy()

    sales_df["dish_id"] = sales_df["dish_id"].astype(str)
    dishes_sales["dish_id"] = dishes_sales["dish_id"].astype(str)

    sales_df = sales_df.merge(
        dishes_sales,
        on="dish_id",
        how="left"
    )

    # ---------------------------------------------------------
    # ADD AREA INFORMATION
    # ---------------------------------------------------------

    areas_sales = data["areas"][
        [
            "area_id",
            "area_name",
            "city"
        ]
    ].copy()

    sales_df["area_id"] = sales_df["area_id"].astype(str)
    areas_sales["area_id"] = areas_sales["area_id"].astype(str)

    sales_df = sales_df.merge(
        areas_sales,
        on="area_id",
        how="left"
    )

    # ---------------------------------------------------------
    # SALES KPIs
    # ---------------------------------------------------------

    total_sales = sales_df["revenue"].sum()
    total_quantity = sales_df["quantity"].sum()
    total_orders = sales_df["order_id"].nunique()

    average_order_value = (
        total_sales / total_orders
        if total_orders > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sales",
        f"₹{total_sales:,.0f}"
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Items Sold",
        f"{total_quantity:,}"
    )

    col4.metric(
        "Average Order Value",
        f"₹{average_order_value:,.2f}"
    )

    st.divider()

    # ---------------------------------------------------------
    # CATEGORY-WISE SALES
    # ---------------------------------------------------------

    st.subheader("Sales by Category")

    category_sales = (
        sales_df
        .dropna(subset=["category"])
        .groupby("category", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum")
        )
        .sort_values(
            "revenue",
            ascending=False
        )
    )

    fig_category = px.bar(
        category_sales,
        x="category",
        y="revenue",
        title="Revenue by Category",
        labels={
            "category": "Category",
            "revenue": "Revenue"
        }
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    st.dataframe(
        category_sales,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # DAILY SALES TREND
    # ---------------------------------------------------------

    st.subheader("Daily Sales Trend")

    daily_sales = (
        sales_df
        .groupby("order_date", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum")
        )
        .sort_values("order_date")
    )

    fig_daily = px.line(
        daily_sales,
        x="order_date",
        y="revenue",
        title="Daily Revenue Trend",
        labels={
            "order_date": "Date",
            "revenue": "Revenue"
        }
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # TOP SELLING DISHES
    # ---------------------------------------------------------

    st.subheader("Top Selling Dishes")

    top_dishes = (
        sales_df
        .dropna(subset=["dish_name"])
        .groupby(
            ["dish_id", "dish_name", "category"],
            as_index=False
        )
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum")
        )
        .sort_values(
            "quantity",
            ascending=False
        )
        .head(10)
    )

    fig_dishes = px.bar(
        top_dishes,
        x="dish_name",
        y="quantity",
        title="Top 10 Selling Dishes",
        labels={
            "dish_name": "Dish",
            "quantity": "Quantity Sold"
        }
    )

    st.plotly_chart(
        fig_dishes,
        use_container_width=True
    )

    st.dataframe(
        top_dishes,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # AREA-WISE SALES
    # ---------------------------------------------------------

    st.subheader("Sales by Area")

    area_sales = (
        sales_df
        .dropna(subset=["area_name"])
        .groupby(
            ["area_id", "area_name", "city"],
            as_index=False
        )
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            orders=("order_id", "nunique")
        )
        .sort_values(
            "revenue",
            ascending=False
        )
    )

    fig_area = px.bar(
        area_sales,
        x="area_name",
        y="revenue",
        title="Revenue by Area",
        labels={
            "area_name": "Area",
            "revenue": "Revenue"
        }
    )

    st.plotly_chart(
        fig_area,
        use_container_width=True
    )

    st.dataframe(
        area_sales,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # PAYMENT METHOD
    # ---------------------------------------------------------

    st.subheader("Sales by Payment Method")

    payment_sales = (
        sales_df
        .groupby(
            "payment_method",
            as_index=False
        )
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique")
        )
        .sort_values(
            "revenue",
            ascending=False
        )
    )

    fig_payment = px.pie(
        payment_sales,
        names="payment_method",
        values="revenue",
        title="Revenue by Payment Method"
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True
    )

    st.dataframe(
        payment_sales,
        use_container_width=True
    )

# ============================================================
# PROFITABILITY
# ============================================================

elif page == "Profitability":

    st.title("📊 Profitability Analysis")

    profit_by_dish = (
        filtered_transactions
        .groupby("dish_id")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum")
        )
        .reset_index()
        .merge(
            data["dishes"][
                ["dish_id", "dish_name", "category"]
            ],
            on="dish_id",
            how="left"
        )
    )

    profit_by_dish["profit_margin"] = (
        profit_by_dish["profit"]
        / profit_by_dish["revenue"]
    )

    profit_by_dish = profit_by_dish.sort_values(
        "profit",
        ascending=False
    )

    fig = px.bar(
        profit_by_dish.head(10),
        x="dish_name",
        y="profit",
        title="Top 10 Most Profitable Dishes"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Profitability Table")

    st.dataframe(
        profit_by_dish,
        use_container_width=True
    )

    st.subheader(
        "Revenue vs Profit"
    )

    fig = px.scatter(
        profit_by_dish,
        x="revenue",
        y="profit",
        size="profit",
        hover_name="dish_name",
        title="Revenue vs Profit by Dish"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DISH PERFORMANCE
# ============================================================

elif page == "Dish Performance":

    st.title("🍽️ Dish Performance")

    dish_summary = (
        filtered_transactions
        .groupby("dish_id")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            orders=("order_id", "nunique")
        )
        .reset_index()
        .merge(
            data["dishes"],
            on="dish_id",
            how="left"
        )
    )

    dish_summary["profit_margin"] = (
        dish_summary["total_profit"]
        / dish_summary["total_revenue"]
    )

    st.subheader("Top Selling Dishes")

    top = dish_summary.sort_values(
        "total_quantity",
        ascending=False
    ).head(10)

    fig = px.bar(
        top,
        x="dish_name",
        y="total_quantity",
        title="Top Selling Dishes"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Low Selling Dishes")

    low = dish_summary.sort_values(
        "total_quantity"
    ).head(10)

    st.dataframe(
        low,
        use_container_width=True
    )


# ============================================================
# AREA PERFORMANCE
# ============================================================

elif page == "Area Performance":

    st.title("📍 Area Performance")

    area_summary = (
        filtered_transactions
        .groupby("area_id")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique"),
            quantity=("quantity", "sum")
        )
        .reset_index()
        .merge(
            data["areas"][
                ["area_id", "area_name", "city"]
            ],
            on="area_id",
            how="left"
        )
    )

    area_summary["profit_margin"] = (
        area_summary["profit"]
        / area_summary["revenue"]
    )

    fig = px.bar(
        area_summary.sort_values(
            "revenue",
            ascending=False
        ),
        x="area_name",
        y="revenue",
        title="Revenue by Area"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Area Performance Table")

    st.dataframe(
        area_summary.sort_values(
            "revenue",
            ascending=False
        ),
        use_container_width=True
    )


# ============================================================
# INVENTORY INTELLIGENCE
# ============================================================

elif page == "Inventory Intelligence":

    st.title("📦 Inventory Intelligence")

    inventory = data["inventory"].copy()

    materials = data["raw_materials"].copy()

    inventory_summary = (
        inventory
        .groupby("material_id")
        .agg(
            purchased=("quantity_purchased", "sum"),
            used=("quantity_used", "sum"),
            wasted=("quantity_wasted", "sum")
        )
        .reset_index()
        .merge(
            materials,
            on="material_id",
            how="left"
        )
    )

    inventory_summary["estimated_stock"] = (
        inventory_summary["purchased"]
        - inventory_summary["used"]
        - inventory_summary["wasted"]
    )

    inventory_summary["wastage_rate"] = np.where(
        inventory_summary["purchased"] > 0,
        inventory_summary["wasted"]
        / inventory_summary["purchased"]
        * 100,
        0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Materials",
        len(inventory_summary)
    )

    col2.metric(
        "Total Wastage",
        f"{inventory_summary['wasted'].sum():,.0f}"
    )

    col3.metric(
        "Avg Wastage Rate",
        f"{inventory_summary['wastage_rate'].mean():.2f}%"
    )

    st.subheader("Material Wastage")

    fig = px.bar(
        inventory_summary.sort_values(
            "wasted",
            ascending=False
        ),
        x="material_name",
        y="wasted",
        title="Material Wastage"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Inventory Details")

    st.dataframe(
        inventory_summary,
        use_container_width=True
    )


# ============================================================
# DEMAND FORECAST
# ============================================================

elif page == "Demand Forecast":

    st.title("🔮 Demand Forecast")

    st.write(
        "XGBoost-based restaurant demand prediction."
    )

    if model is None:

        st.error(
            "Demand model not found. "
            "Run python src\\train_model.py first."
        )

    else:

        st.success(
            "XGBoost demand model loaded successfully."
        )

        latest = (
            ml_data
            .sort_values("order_date")
            .groupby(
                ["area_id", "dish_id"],
                as_index=False
            )
            .tail(1)
            .copy()
        )

        feature_columns = encoders.get(
            "features",
            []
        )

        if feature_columns:

            X_forecast = latest[
                feature_columns
            ].copy()

            area_encoder = encoders.get(
                "area_encoder",
                {}
            )

            dish_encoder = encoders.get(
                "dish_encoder",
                {}
            )

            X_forecast["area_id"] = (
                X_forecast["area_id"]
                .astype(str)
                .map(area_encoder)
            )

            X_forecast["dish_id"] = (
                X_forecast["dish_id"]
                .astype(str)
                .map(dish_encoder)
            )

            X_forecast = X_forecast.replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

            predictions = model.predict(
                X_forecast
            )

            latest["predicted_demand"] = np.maximum(
                predictions,
                0
            )

            latest = latest.merge(
                data["dishes"][
                    ["dish_id", "dish_name"]
                ],
                on="dish_id",
                how="left"
            )

            st.subheader(
                "Predicted Demand by Dish"
            )

            forecast_summary = (
                latest
                .groupby("dish_name")
                ["predicted_demand"]
                .mean()
                .reset_index()
                .sort_values(
                    "predicted_demand",
                    ascending=False
                )
            )

            fig = px.bar(
                forecast_summary.head(10),
                x="dish_name",
                y="predicted_demand",
                title="Top Predicted Demand"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                forecast_summary,
                use_container_width=True
            )


# ============================================================
# ML PERFORMANCE
# ============================================================

elif page == "ML Performance":

    st.title("🤖 Machine Learning Performance")

    if metrics:

        validation = metrics.get(
            "validation",
            {}
        )

        test = metrics.get(
            "test",
            {}
        )

        st.subheader("Validation Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "MAE",
            f"{validation.get('MAE', 0):.4f}"
        )

        col2.metric(
            "RMSE",
            f"{validation.get('RMSE', 0):.4f}"
        )

        col3.metric(
            "MAPE",
            f"{validation.get('MAPE', 0):.2f}%"
        )

        col4.metric(
            "R²",
            f"{validation.get('R2', 0):.4f}"
        )

        st.subheader("Test Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "MAE",
            f"{test.get('MAE', 0):.4f}"
        )

        col2.metric(
            "RMSE",
            f"{test.get('RMSE', 0):.4f}"
        )

        col3.metric(
            "MAPE",
            f"{test.get('MAPE', 0):.2f}%"
        )

        col4.metric(
            "R²",
            f"{test.get('R2', 0):.4f}"
        )

        st.info(
            "The model currently explains approximately "
            "31% of demand variation on the test set. "
            "MAPE is relatively high, so forecasts should "
            "be treated as decision-support estimates."
        )

    else:

        st.warning(
            "Model metrics file not found."
        )


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "Business Recommendations":

    st.title("💡 Business Recommendations")

    st.markdown(
        "### Automatically generated recommendations"
    )

    dish_summary = (
        filtered_transactions
        .groupby("dish_id")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum")
        )
        .reset_index()
        .merge(
            data["dishes"][
                ["dish_id", "dish_name"]
            ],
            on="dish_id",
            how="left"
        )
    )

    inventory = data["inventory"]

    materials = data["raw_materials"]

    inventory_summary = (
        inventory
        .groupby("material_id")
        .agg(
            purchased=("quantity_purchased", "sum"),
            wasted=("quantity_wasted", "sum")
        )
        .reset_index()
        .merge(
            materials[
                ["material_id", "material_name"]
            ],
            on="material_id",
            how="left"
        )
    )

    inventory_summary["wastage_rate"] = (
        inventory_summary["wasted"]
        / inventory_summary["purchased"]
        * 100
    )

    top_dish = dish_summary.sort_values(
        "total_quantity",
        ascending=False
    ).iloc[0]

    profitable_dish = dish_summary.sort_values(
        "total_profit",
        ascending=False
    ).iloc[0]

    wastage_material = inventory_summary.sort_values(
        "wastage_rate",
        ascending=False
    ).iloc[0]

    st.success(
        f"🏆 Increase inventory planning for "
        f"**{top_dish['dish_name']}** because it is "
        f"the highest-volume dish."
    )

    st.success(
        f"💰 Promote **{profitable_dish['dish_name']}** "
        f"because it generates the highest total profit."
    )

    st.warning(
        "⚠️ Review pricing, discounts and ingredient "
        "costs for high-sales / low-profit dishes."
    )

    st.warning(
        f"🗑️ Investigate wastage of "
        f"**{wastage_material['material_name']}** "
        f"because it has the highest wastage rate."
    )

    st.info(
        "📈 Use demand forecasts to improve purchasing "
        "and kitchen preparation planning."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Restaurant Sales Intelligence Platform"
)

st.sidebar.caption(
    "Python • Pandas • XGBoost • Streamlit • Plotly"
)