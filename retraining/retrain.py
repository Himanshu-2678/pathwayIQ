import os
import pandas as pd
import joblib
import logging

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# =========================================
# LOGGING CONFIGURATION
# =========================================

logging.basicConfig(
    filename="logs/retraining.log",
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger(__name__)


# =========================================
# LOAD DATA
# =========================================

print("Loading dataset...")

df = pd.read_csv(
    "dataset/processed/cleaned_diabetic_data.csv"
)

print(f"Dataset shape: {df.shape}")


# =========================================
# CREATE REFERENCE / CURRENT DATA
# =========================================

reference_data, current_data = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# =========================================
# SIMULATE PRODUCTION DRIFT
# =========================================

print("Simulating production drift...")


current_data["number_emergency"] = (
    current_data["number_emergency"] + 3
).clip(upper=20)

current_data["number_inpatient"] = (
    current_data["number_inpatient"] + 2
).clip(upper=20)

current_data["time_in_hospital"] = (
    current_data["time_in_hospital"] + 4
).clip(upper=30)

current_data["number_diagnoses"] = (
    current_data["number_diagnoses"] + 3
).clip(upper=20)


# =========================================
# MANUAL DRIFT DETECTION
# =========================================

print("\nCalculating drift metrics...")


drift_features = []


monitor_columns = [
    "number_emergency",
    "number_inpatient",
    "time_in_hospital",
    "number_diagnoses"
]


for column in monitor_columns:

    reference_mean = reference_data[column].mean()

    current_mean = current_data[column].mean()

    percent_shift = abs(
        current_mean - reference_mean
    ) / reference_mean


    print(f"\n{column}")
    print(f"Reference Mean: {reference_mean:.2f}")
    print(f"Current Mean: {current_mean:.2f}")
    print(f"Shift: {percent_shift:.2f}")


    # Drift threshold for each feature
    if percent_shift > 0.30:

        drift_features.append(column)


# =========================================
# DRIFT SUMMARY
# =========================================

drifted_features = len(drift_features)

total_features = len(monitor_columns)

drift_ratio = drifted_features / total_features

dataset_drift = drift_ratio >= 0.50


print("\n=================================")
print(f"Dataset Drift Detected: {dataset_drift}")
print(f"Drifted Features: {drifted_features}/{total_features}")
print(f"Drift Ratio: {drift_ratio:.2f}")
print("=================================")


# =========================================
# RETRAINING CONDITION
# =========================================

DRIFT_THRESHOLD = 0.30


if drift_ratio >= DRIFT_THRESHOLD:

    print("\nDrift threshold exceeded.")
    print("Retraining model...")


    # =====================================
    # COMBINE HISTORICAL + CURRENT DATA
    # =====================================

    retraining_data = pd.concat([
        reference_data,
        current_data
    ])


    X_train = retraining_data.drop(
        columns=["target"]
    )

    y_train = retraining_data["target"]


    # =====================================
    # COLUMN TYPES
    # =====================================

    categorical_cols = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_cols = X_train.select_dtypes(
        exclude="object"
    ).columns.tolist()


    # =====================================
    # PREPROCESSING
    # =====================================

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )


    # =====================================
    # HANDLE CLASS IMBALANCE
    # =====================================

    negative_count = (y_train == 0).sum()

    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )


    # =====================================
    # RETRAIN MODEL
    # =====================================

    retrained_pipeline = Pipeline(steps=[

        ("preprocessor", preprocessor),

        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ))
    ])


    retrained_pipeline.fit(
        X_train,
        y_train
    )

    print("Retraining completed.")


    # =====================================
    # MODEL VERSIONING
    # =====================================

    versions_dir = ("models/versions")

    existing_models = [file for file in os.listdir(versions_dir) if file.startswith("xgb_model_v")]

    new_version = len(existing_models) + 1

    model_filename = (f"xgb_model_v{new_version}.pkl")

    model_path = os.path.join(versions_dir, model_filename)

    joblib.dump(retrained_pipeline, model_path)

    print(f"Retrained model saved: {model_path}")


    # =====================================
    # LOG RETRAIN EVENT
    # =====================================

    retrain_log = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event": "MODEL_RETRAINED",
        "model_version": new_version,
        "model_filename": model_filename,
        "drift_ratio": round(drift_ratio, 4),
        "drifted_features": drifted_features,
        "drift_columns": drift_features,
        "training_rows": len(retraining_data),
        "model_path": model_path
    }

    logger.info(retrain_log)

    print("\nRetraining event logged.")


else:

    print("\nDrift threshold not exceeded.")
    print("Retraining skipped.")
