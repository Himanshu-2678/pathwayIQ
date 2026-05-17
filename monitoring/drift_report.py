import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset


# =========================================
# LOAD DATA
# =========================================

print("Loading dataset...")

df = pd.read_csv("dataset\processed\cleaned_diabetic_data.csv")

print(f"Dataset shape: {df.shape}")


# =========================================
# SPLIT REFERENCE / CURRENT DATA
# =========================================

reference_data, current_data = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# =========================================
# SIMULATE REAL-WORLD DRIFT
# =========================================

print("Simulating hospital population drift...")


# Increase emergency visits
current_data["number_emergency"] = (
    current_data["number_emergency"] + 3
).clip(upper=20)


# Increase inpatient admissions
current_data["number_inpatient"] = (
    current_data["number_inpatient"] + 2
).clip(upper=20)


# Increase hospital stay duration
current_data["time_in_hospital"] = (
    current_data["time_in_hospital"] + 4
).clip(upper=30)


# Increase diagnosis complexity
current_data["number_diagnoses"] = (
    current_data["number_diagnoses"] + 3
).clip(upper=20)

print("Reference shape:", reference_data.shape)
print("Current shape:", current_data.shape)


# =========================================
# CREATE DRIFT REPORT
# =========================================

print("Generating Evidently drift report...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

evaluation = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# =========================================
# SAVE REPORT
# =========================================

evaluation.save_html("drift_report.html")

print("\nDrift report generated successfully.")
print("Saved as: drift_report.html")