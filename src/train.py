import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier


# =========================================
# LOAD DATA
# =========================================

print("Loading dataset...")

df = pd.read_csv(r"dataset\processed\cleaned_diabetic_data.csv")

X = df.drop(columns=["target"])
y = df["target"]

print(f"Dataset shape: {df.shape}")


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train-test split completed.")


# =========================================
# COLUMN TYPES
# =========================================

categorical_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns

numerical_cols = X_train.select_dtypes(
    exclude="object"
).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")


# =========================================
# PREPROCESSING
# =========================================

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

print("Preprocessing pipeline created.")


# =========================================
# CLASS IMBALANCE HANDLING
# =========================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print(f"scale_pos_weight: {scale_pos_weight:.2f}")


# =========================================
# MODEL PIPELINE
# =========================================

model_pipeline = Pipeline(steps=[
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

print("Training XGBoost model...")


# =========================================
# TRAIN MODEL
# =========================================

model_pipeline.fit(X_train, y_train)

print("Model training completed.")


# =========================================
# EVALUATION
# =========================================

y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)

print(f"ROC-AUC Score: {roc_auc:.4f}")


# =========================================
# SAVE MODEL
# =========================================

joblib.dump(model_pipeline, r"models\xgb_readmission_pipeline.pkl")

print("\nModel saved successfully.")