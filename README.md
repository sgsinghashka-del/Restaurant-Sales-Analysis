# Restaurant Sales Analysis & Intelligence Platform

An end-to-end Data Science and Business Intelligence project for analyzing restaurant sales, profitability, dish performance, area performance, inventory wastage, and demand trends.

The project combines **Python, Pandas, SQL, XGBoost, Streamlit, SQLite, and Pytest** to transform raw restaurant transaction data into actionable business insights.

---

## 1. Project Overview

Restaurant businesses generate large amounts of transactional, sales, inventory, and operational data.

This project provides an analytics platform that helps restaurant management answer questions such as:

* Which dishes generate the highest sales?
* Which dishes generate the highest profit?
* Which products have high sales but relatively low profitability?
* Which restaurant areas generate the most revenue?
* Which categories perform best?
* Which raw materials have high wastage?
* Which materials need inventory attention?
* How are sales changing over time?
* Can future dish demand be predicted?
* Where should management focus marketing and inventory planning?

The platform provides both **historical business analytics** and **machine-learning-based demand prediction**.

---

## 2. Business Problem

Restaurant management needs to continuously monitor:

* Sales performance
* Profitability
* Customer demand
* Dish performance
* Area performance
* Inventory usage
* Raw-material wastage
* Supplier-related operations
* Future demand

Manually analyzing these areas from transaction data is time-consuming and makes it difficult to identify trends.

The objective of this project is to build a centralized analytics solution that converts restaurant operational data into measurable business insights.

---

## 3. Project Objectives

The main objectives are:

1. Analyze restaurant sales and transaction performance.
2. Identify top-selling and low-selling dishes.
3. Analyze dish-level profitability.
4. Identify high-sales but low-profit products.
5. Compare restaurant performance by geographic area.
6. Analyze category-level sales and profitability.
7. Analyze payment methods and order trends.
8. Measure raw-material usage and wastage.
9. Identify inventory items requiring attention.
10. Build a demand prediction model using XGBoost.
11. Provide an interactive Streamlit dashboard.
12. Implement SQL-based business analytics.
13. Add automated testing using Pytest.
14. Prepare the project for deployment and production-style usage.

---

## 4. Dataset

The project contains multiple related datasets.

### Areas

Contains restaurant operating areas and cities.

Columns include:

* area_id
* area_name
* city

### Restaurants

Contains restaurant-level information.

### Dishes

Contains menu item information.

Columns:

* dish_id
* dish_name
* category
* selling_price
* ingredient_cost
* demand_weight

### Orders

Contains customer order-level information.

Columns:

* order_id
* order_datetime
* order_date
* customer_id
* restaurant_id
* area_id
* order_value
* discount
* payment_method
* order_status
* promotion

### Order Items

Contains dish-level transaction information.

Columns:

* order_item_id
* order_id
* dish_id
* quantity
* selling_price
* discount
* revenue
* ingredient_cost
* profit

### Raw Materials

Contains inventory raw-material information.

Columns:

* material_id
* material_name
* category
* shelf_life_days
* unit_cost
* reorder_level
* supplier_id

### Inventory

Contains material purchasing, usage, and wastage information.

---

## 5. Dataset Size

The current dataset contains:

| Dataset       |    Rows |
| ------------- | ------: |
| Areas         |       8 |
| Restaurants   |       8 |
| Dishes        |      18 |
| Raw Materials |      15 |
| Suppliers     |       8 |
| Orders        | 152,963 |
| Order Items   | 223,379 |
| Inventory     |     720 |

Total records across datasets:

**377,119**

---

## 6. Overall Business Performance

The transaction analysis produced the following results:

| Metric          |          Value |
| --------------- | -------------: |
| Unique Orders   |        152,963 |
| Items Sold      |        303,405 |
| Revenue         |    ₹75,795,023 |
| Ingredient Cost |    ₹29,908,289 |
| Profit          | ₹36,366,342.76 |
| Profit Margin   |         47.98% |

The results provide a baseline for evaluating restaurant sales and profitability.

---

## 7. Key Business Insights

### Top Selling Dishes

The highest-volume dishes include:

1. Paneer Tikka
2. Chicken Biryani
3. Butter Chicken
4. Veg Biryani
5. Margherita Pizza
6. Dal Makhani
7. Hakka Noodles
8. Masala Dosa
9. Farmhouse Pizza
10. Manchurian

Paneer Tikka is the highest-volume dish with **24,040 units sold**.

---

### Most Profitable Dishes

The highest-profit dishes include:

* Farmhouse Pizza
* Margherita Pizza
* Paneer Tikka
* Chicken Biryani
* Butter Chicken

Farmhouse Pizza generated approximately **₹3.40 million in total profit**, making it the highest-profit dish in the analysis.

---

### High-Sales / Low-Profit Analysis

The project specifically identifies dishes that generate significant sales volume but comparatively lower profit.

This analysis can help management review:

* Selling price
* Discounts
* Ingredient costs
* Portion sizes
* Supplier costs
* Promotional strategies

---

### Area Performance

The highest-revenue areas include:

1. Andheri West — Mumbai
2. Connaught Place — Delhi
3. Koramangala — Bengaluru
4. Banjara Hills — Hyderabad
5. Hinjewadi — Pune

Andheri West generated approximately **₹12.11 million in revenue**.

---

### Category Performance

Major categories analyzed include:

* Pizza
* Rice & Biryani
* Starters
* Main Course
* Chinese
* Fast Food
* Beverages
* Dessert
* South Indian
* North Indian

Pizza generated the highest category revenue at approximately **₹14.34 million**.

---

## 8. Inventory Intelligence

The inventory module analyzes:

* Quantity purchased
* Quantity used
* Quantity wasted
* Unit cost
* Reorder levels
* Material categories
* Supplier relationships

The system also calculates material wastage rates.

### High-Wastage Materials

Examples include:

* Cocoa Powder
* Tomato
* Basmati Rice
* Bread
* Paneer

Cocoa Powder has one of the highest observed wastage rates at approximately **8.16%**.

---

## 9. Machine Learning

### Model

The project uses **XGBoost** for restaurant dish-demand prediction.

The model predicts demand using features related to:

* Area
* Dish
* Date
* Day of week
* Month
* Weekend indicator
* Discounts
* Promotions
* Historical demand
* Rolling demand
* Revenue trends
* Profit trends
* Demand growth

### Feature Engineering

A total of **21 machine-learning features** were created.

Important features include:

* lag_1
* lag_7
* lag_14
* lag_30
* rolling_7
* rolling_14
* rolling_30
* revenue_rolling_7
* profit_rolling_7
* demand_trend
* demand_growth

### Dataset

The engineered ML dataset contains:

**103,835 rows**

with demand values ranging from:

**1 to 18 units**

---

## 10. Model Performance

### Validation

| Metric | Result |
| ------ | -----: |
| MAE    | 1.1502 |
| RMSE   | 1.5658 |
| MAPE   | 49.58% |
| R²     | 0.3131 |

### Test

| Metric | Result |
| ------ | -----: |
| MAE    | 1.0309 |
| RMSE   | 1.3818 |
| MAPE   | 51.04% |
| R²     | 0.3092 |

The model provides a baseline demand forecasting capability that can support inventory and operational planning.

---

## 11. Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Dashboard sections:

1. Executive Dashboard
2. Sales Analysis
3. Profitability
4. Dish Performance
5. Area Performance
6. Inventory In

🚀 Live Demo
[[View Live Dashboard]](https://restaurant-sales-analysis-x2jjqzhnjzw43qckzrgopm.streamlit.app/)

💻 GitHub Repository
[View Source Code]https://github.com/sgsinghashka-del/Restaurant-Sales-Analysis/tree/main


