from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import shap
import numpy as np
from src.predict import clean_feature_name
from src.thresholding import assign_risk_band

import logging
import uuid
import time
from datetime import datetime

from src.calibration_utils import load_calibrated_model

## Load model
# Raw pipeline for SHAP explainability
model = joblib.load("models/xgb_readmission_pipeline.pkl")

# Calibrated pipeline for probability inference
calibrated_model = load_calibrated_model()

xgb_model = model.named_steps["classifier"]

explainer = shap.TreeExplainer(xgb_model)

print("Model loaded successfully")


app = FastAPI(
    title="PathwayIQ Readmission API"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger(__name__)

class PatientData(BaseModel):

    age: int = Field(..., ge=0, le=9)
    time_in_hospital: int = Field(..., ge=1, le=30)
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int = Field(..., ge=0, le=50)
    number_inpatient: int = Field(..., ge=0, le=50)
    number_diagnoses: int = Field(..., ge=1, le=20)

    race: str
    gender: Literal["Male", "Female"]
    payer_code: str
    medical_specialty: str

    admission_type_id: str
    discharge_disposition_id: str
    admission_source_id: str

    diag_1: str
    diag_2: str
    diag_3: str

    max_glu_serum: str
    A1Cresult: str

    metformin: str
    repaglinide: str
    nateglinide: str
    glimepiride: str
    glipizide: str
    glyburide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    insulin: Literal["No", "Steady", "Up", "Down"]

    glipizide_metformin: str
    glyburide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str

    change: str
    diabetesMed: Literal["Yes", "No"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# PREDICTION ENDPOINT
@app.post("/predict")
def predict(data: PatientData):

    request_id = str(uuid.uuid4())

    start_time = time.time()

    # =========================================
    # CREATE INPUT DATAFRAME
    # =========================================

    input_df = pd.DataFrame([data.dict()])

    input_df.rename(columns={
        "glipizide_metformin": "glipizide-metformin",
        "glyburide_metformin": "glyburide-metformin",
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone"
    }, inplace=True)

    # =========================================
    # PREDICTION
    # =========================================

    probability = calibrated_model.predict_proba(input_df)[0][1]
    risk_label = assign_risk_band(probability)

    # =========================================
    # TRANSFORM INPUT
    # =========================================

    transformed_input = model.named_steps[
        "preprocessor"
    ].transform(input_df)

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    # =========================================
    # SHAP VALUES
    # =========================================

    shap_values = explainer.shap_values(transformed_input)

    shap_row = shap_values[0]

    # =========================================
    # TOP FEATURES
    # =========================================

    top_indices = np.argsort(np.abs(shap_row))[::-1][:5]

    top_factors = []

    for idx in top_indices:

        feature_name = clean_feature_name(
            feature_names[idx]
        )

        contribution = float(shap_row[idx])

        top_factors.append({
            "feature": feature_name,
            "impact": round(contribution, 4)
        })

    # =========================================
    # RESPONSE
    # =========================================

    latency_ms = round(
        (time.time() - start_time) * 1000,
        2
    )

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "risk_score": round(float(probability), 4),
        "risk_label": risk_label,
        "latency_ms": latency_ms
    }

    logger.info(log_data)

    return {
        "request_id": request_id,
        "risk_score": round(float(probability), 4),
        "risk_label": risk_label,
        "latency_ms": latency_ms,
        "top_risk_factors": top_factors
    }   