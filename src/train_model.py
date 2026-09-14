import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

from data_processing import (
    load_data,
    prepare_transactions
)

from feature_engineering import (
    create_ml_dataset,
    get_feature_columns
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAPE FUNCTION
# ============================================================

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.

    Zero actual values are ignored to avoid
    division-by-zero problems.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return 0.0

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("       RESTAURANT DEMAND MODEL")
print("========================================")

print("\n1. Loading datasets...")

data = load_data(DATA_DIR)

print("Datasets loaded successfully.")


# ============================================================
# PREPARE TRANSACTIONS
# ============================================================

print("\n2. Preparing transactions...")

transactions = prepare_transactions(data)

print(
    f"Transaction rows: {len(transactions):,}"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n3. Creating machine learning dataset...")

ml_data = create_ml_dataset(
    transactions
)

print(
    f"ML rows: {len(ml_data):,}"
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

features = get_feature_columns()

target = "demand"

X = ml_data[features].copy()

y = ml_data[target].copy()


# ============================================================
# ENCODE CATEGORICAL IDs
# ============================================================

print("\n4. Encoding area and dish IDs...")

# Area mapping
area_values = sorted(
    ml_data["area_id"]
    .astype(str)
    .unique()
)

area_encoder = {
    value: index
    for index, value in enumerate(area_values)
}

# Dish mapping
dish_values = sorted(
    ml_data["dish_id"]
    .astype(str)
    .unique()
)

dish_encoder = {
    value: index
    for index, value in enumerate(dish_values)
}


# Apply encodings
X["area_id"] = (
    X["area_id"]
    .astype(str)
    .map(area_encoder)
)

X["dish_id"] = (
    X["dish_id"]
    .astype(str)
    .map(dish_encoder)
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n5. Creating time-based train/validation/test split...")

ml_data = ml_data.sort_values(
    "order_date"
).reset_index(drop=True)

X = X.loc[
    ml_data.index
].reset_index(drop=True)

y = y.loc[
    ml_data.index
].reset_index(drop=True)


# Date boundaries
dates = ml_data["order_date"]

train_cutoff = dates.quantile(0.75)

validation_cutoff = dates.quantile(0.85)


train_mask = (
    dates <= train_cutoff
)

validation_mask = (
    (dates > train_cutoff)
    &
    (dates <= validation_cutoff)
)

test_mask = (
    dates > validation_cutoff
)


X_train = X.loc[
    train_mask
].copy()

y_train = y.loc[
    train_mask
].copy()


X_validation = X.loc[
    validation_mask
].copy()

y_validation = y.loc[
    validation_mask
].copy()


X_test = X.loc[
    test_mask
].copy()

y_test = y.loc[
    test_mask
].copy()


print(
    "\nTrain rows:",
    f"{len(X_train):,}"
)

print(
    "Validation rows:",
    f"{len(X_validation):,}"
)

print(
    "Test rows:",
    f"{len(X_test):,}"
)

print(
    "\nTrain end:",
    train_cutoff
)

print(
    "Validation end:",
    validation_cutoff
)


# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

print("\n6. Training XGBoost model...")

model = XGBRegressor(

    n_estimators=400,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=2
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_validation,
            y_validation
        )
    ],

    verbose=False
)


print(
    "XGBoost training completed."
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

print("\n7. Evaluating validation data...")

validation_predictions = model.predict(
    X_validation
)

validation_predictions = np.maximum(
    validation_predictions,
    0
)


validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)

validation_mape = calculate_mape(
    y_validation,
    validation_predictions
)


# ============================================================
# TEST PREDICTION
# ============================================================

print("\n8. Evaluating test data...")

test_predictions = model.predict(
    X_test
)

test_predictions = np.maximum(
    test_predictions,
    0
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

test_mape = calculate_mape(
    y_test,
    test_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("       MODEL PERFORMANCE")
print("========================================")

print("\nVALIDATION")

print(
    f"MAE  : {validation_mae:.4f}"
)

print(
    f"RMSE : {validation_rmse:.4f}"
)

print(
    f"MAPE : {validation_mape:.2f}%"
)

print(
    f"R²   : {validation_r2:.4f}"
)


print("\nTEST")

print(
    f"MAE  : {test_mae:.4f}"
)

print(
    f"RMSE : {test_rmse:.4f}"
)

print(
    f"MAPE : {test_mape:.2f}%"
)

print(
    f"R²   : {test_r2:.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n9. Calculating feature importance...")

importance = pd.DataFrame({

    "feature": features,

    "importance": model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


print("\nTop 10 important features:")

print(
    importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "xgboost_demand_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print(
    f"\nModel saved to: {model_path}"
)


# ============================================================
# SAVE ENCODERS
# ============================================================

encoders = {

    "area_encoder": area_encoder,

    "dish_encoder": dish_encoder,

    "features": features
}


encoder_path = (
    MODEL_DIR /
    "encoders.json"
)


with open(
    encoder_path,
    "w"
) as file:

    json.dump(
        encoders,
        file,
        indent=4
    )


print(
    f"Encoders saved to: {encoder_path}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "validation": {

        "MAE": float(validation_mae),

        "RMSE": float(validation_rmse),

        "MAPE": float(validation_mape),

        "R2": float(validation_r2)
    },

    "test": {

        "MAE": float(test_mae),

        "RMSE": float(test_rmse),

        "MAPE": float(test_mape),

        "R2": float(test_r2)
    },

    "dataset": {

        "total_rows": int(len(ml_data)),

        "train_rows": int(len(X_train)),

        "validation_rows": int(
            len(X_validation)
        ),

        "test_rows": int(
            len(X_test)
        )
    }
}


metrics_path = (
    MODEL_DIR /
    "metrics.json"
)


with open(
    metrics_path,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    f"Metrics saved to: {metrics_path}"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = ml_data.loc[
    test_mask
].copy()

test_results = test_results.reset_index(
    drop=True
)

test_results["actual_demand"] = (
    y_test
    .reset_index(drop=True)
)

test_results["predicted_demand"] = (
    test_predictions
)

test_results["prediction_error"] = (
    test_results["actual_demand"]
    -
    test_results["predicted_demand"]
)


prediction_path = (
    MODEL_DIR /
    "test_predictions.csv"
)


test_results.to_csv(
    prediction_path,
    index=False
)


print(
    f"Test predictions saved to: {prediction_path}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("       MODEL TRAINING COMPLETE")
print("========================================")

print("\nGenerated files:")

print(
    "1.",
    model_path
)

print(
    "2.",
    encoder_path
)

print(
    "3.",
    metrics_path
)

print(
    "4.",
    prediction_path
)

print(
    "\nDemand prediction model is ready."
)