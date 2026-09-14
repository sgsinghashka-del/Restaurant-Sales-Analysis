from pathlib import Path
import sys
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_demand_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoders.json"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "models" / "test_predictions.csv"


def test_model_file_exists():

    assert MODEL_PATH.exists(), (
        f"Model file not found: {MODEL_PATH}"
    )


def test_model_can_be_loaded():

    model = joblib.load(MODEL_PATH)

    assert model is not None

    assert hasattr(model, "predict"), (
        "Loaded model does not have a predict() method"
    )


def test_encoders_file_exists():

    assert ENCODER_PATH.exists(), (
        f"Encoder file not found: {ENCODER_PATH}"
    )

    with open(ENCODER_PATH, "r") as file:
        encoders = json.load(file)

    assert isinstance(encoders, dict)


def test_metrics_file_is_valid():

    assert METRICS_PATH.exists(), (
        f"Metrics file not found: {METRICS_PATH}"
    )

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    assert isinstance(metrics, dict)

    assert len(metrics) > 0


def test_test_predictions_exist():

    assert PREDICTIONS_PATH.exists(), (
        f"Prediction file not found: {PREDICTIONS_PATH}"
    )

    predictions = pd.read_csv(PREDICTIONS_PATH)

    assert len(predictions) > 0

    assert predictions.shape[1] >= 2