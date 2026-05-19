import joblib
from pathlib import Path


CALIBRATED_MODEL_PATH = Path("models/xgb_calibrated_pipeline.pkl")


def load_calibrated_model():
    """
    Load calibrated inference pipeline.
    """

    return joblib.load(CALIBRATED_MODEL_PATH)