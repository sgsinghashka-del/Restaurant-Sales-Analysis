# Restaurant Sales Analysis & Intelligence Platform

<p align="center">
  <img src="Business%20recomtion.png" alt="Restaurant Sales Intelligence Dashboard" width="900" />
</p>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Plotly-Interactive-3B82F6?logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/XGBoost-Demand%20Forecasting-4B0082?logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white" alt="SQLite" />
</div>

An end-to-end data science and business intelligence project for analyzing restaurant sales, inventory performance, dish profitability, area performance, and future demand.

This project turns raw restaurant operational data into actionable business intelligence through interactive dashboards, SQL-based analytics, and machine learning-driven forecasting.

## Overview

Restaurant businesses generate large volumes of transaction, sales, inventory, and customer behavior data. Managing this information manually is inefficient and often hides the real drivers behind profitability and waste.

This platform helps restaurant teams answer critical business questions:

- Which dishes generate the highest sales and profit?
- Which products are high-volume but low-margin?
- Which areas contribute the most revenue?
- Where is inventory waste increasing?
- What will demand likely look like next period?
- Which business decisions should be prioritized?

## Key Features

- Executive sales and profitability dashboard
- Dish-level performance analysis
- Area-wise revenue and trend tracking
- Inventory and wastage intelligence
- Business recommendation engine
- XGBoost-based demand forecasting
- Interactive visual analytics with Streamlit + Plotly
- Automated testing with Pytest

## Business Impact

The platform helps decision-makers improve:

- Menu optimization
- Pricing and discount strategy
- Inventory purchasing and stock control
- Supplier and replenishment planning
- Area-specific operational decisions
- Forecast-driven kitchen preparation planning

## Dashboard Preview

### Executive & Sales Overview

<table>
  <tr>
    <td><img src="Excutive%20dashboard.png" alt="Executive Dashboard" width="480" /></td>
    <td><img src="sales%20board.png" alt="Sales Board" width="480" /></td>
  </tr>
</table>

### Performance Analysis

<table>
  <tr>
    <td><img src="dish%20performance.png" alt="Dish Performance" width="480" /></td>
    <td><img src="area%20performance.png" alt="Area Performance" width="480" /></td>
  </tr>
</table>

### Inventory & Intelligence

<table>
  <tr>
    <td><img src="inventoy%20.png" alt="Inventory Dashboard" width="480" /></td>
    <td><img src="Business%20recomtion.png" alt="Business Recommendations" width="480" /></td>
  </tr>
</table>

## Architecture

<p align="center">
  <img src="Architecture%20Diagram" alt="System architecture diagram" width="1000" />
</p>

The solution combines:

- Data ingestion and cleaning from structured restaurant datasets
- SQL/analytics layer for business aggregation
- Feature engineering for ML forecasting
- Streamlit dashboard for interactive exploration
- Model evaluation and recommendations for managerial decisions

## Project Structure

```text
Restaurant-Sales-Analysis/
├── app/
│   └── app.py
├── data/
│   └── restaurant datasets
├── database/
├── models/
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── ...
├── tests/
├── .streamlit/
├── Architecture Diagram
├── Business recomtion.png
├── Excutive dashboard.png
├── area performance.png
├── dish performance.png
├── inventoy .png
├── sales board.png
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Tech Stack

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- XGBoost
- scikit-learn
- SQLite
- Pytest

## Data Scope

The project analyzes multiple restaurant data domains, including:

- Areas and restaurant locations
- Dishes and categories
- Orders and order items
- Payment methods and promotions
- Inventory and raw materials
- Supplier and wastage data

## Example Insights

The analysis typically reveals:

- Highest profit-generating menu items
- Top-selling dishes by volume
- Area-level revenue leaders
- Material wastage hotspots
- Menu items with strong demand but weak profitability
- Opportunities for pricing and inventory adjustments

## Machine Learning Forecasting

The project includes an XGBoost model for demand prediction using historical sales signals and feature engineering such as:

- lag features
- rolling trends
- revenue and profit trends
- discount and promotion variables
- area and dish information
- date-based features

This supports better forecasting for demand planning and kitchen operations.

## Live Demo

- Streamlit App: https://restaurant-sales-analysis-x2jjqzhnjzw43qckzrgopm.streamlit.app/
- GitHub Repository: https://github.com/sgsinghashka-del/Restaurant-Sales-Analysis

## Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/sgsinghashka-del/Restaurant-Sales-Analysis.git
cd Restaurant-Sales-Analysis
```

2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the dashboard

```bash
streamlit run app/app.py
```

5. Optional: train the model

```bash
python src/train_model.py
```

## Model Performance Snapshot

The model provides a baseline forecasting capability for restaurant demand planning.

| Metric | Value |
|--------|-------|
| MAE | 1.03 |
| RMSE | 1.38 |
| MAPE | 51.04% |
| R² | 0.3092 |

## Business Recommendations

The platform automatically highlights recommendations around:

- Inventory planning for high-volume dishes
- Profit optimization for top-performing items
- Pricing review for low-margin but popular menu items
- Waste reduction opportunities
- Demand-driven purchasing decisions

## License

This project is licensed under the MIT License.

## Contributing

Contributions, feature requests, and pull requests are welcome. If you want to improve forecasting logic, dashboard UX, or analytics depth, feel free to open an issue or submit a PR.

## Contact

Project repository: https://github.com/sgsinghashka-del/Restaurant-Sales-Analysis

---

<p align="center">
  <strong>Built for smarter restaurant insights and better business decisions.</strong>
</p>
