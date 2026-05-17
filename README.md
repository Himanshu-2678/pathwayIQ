# PathwayIQ - Explainable Hospital Readmission Intelligence System

## Live Demo

**Public Deployment URL:** http://pathwayiq.duckdns.org


PathwayIQ is an end-to-end machine learning system designed to predict 30-day hospital readmission risk for diabetic patients using clinical and encounter-level healthcare data.

The project goes beyond traditional model training by incorporating explainable AI with SHAP, drift detection and monitoring, automated retraining triggers, versioned ML artifacts, prediction logging and observability, FastAPI inference serving, and a product-style frontend interface. The emphasis is on building a realistic AI system rather than only training a model.



## System Architecture

![Architecture Diagram](assets/pathwayIQ-architecture-diagram.png)


## Problem Statement

Hospital readmissions are a major challenge in healthcare systems due to increased operational costs, resource strain, patient health risks, and inefficient discharge planning. The objective of PathwayIQ is to identify high-risk patients before discharge using interpretable machine learning.

Dataset Link: [UCI-Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Key Features

**Explainable AI Predictions**   
XGBoost-based readmission prediction with SHAP-powered local feature explanations. Top contributing risk factors are returned per prediction.

**End-to-End ML Pipeline**   
Covers preprocessing, feature engineering, categorical encoding, imbalance handling, and pipeline serialization.

**FastAPI Backend**   
REST inference API with input validation, structured prediction responses, and latency tracking.

**Product-Style Frontend**   
Clean healthcare-focused UI with real-time risk scoring, visual prediction interface, and risk explanation display.

**Observability and Logging**   
Prediction logging, retraining event logging, request IDs, and inference latency tracking.

**Drift Detection**   
Simulated production drift with distribution shift monitoring and threshold-based drift analysis.

**Retraining Pipeline**   
Automatic retraining trigger logic, versioned model artifacts, and lifecycle-aware model management.


## Quantified System Highlights

- Trained on 100K+ real-world diabetic patient encounters
- Processes 39 engineered clinical and encounter-level features
- Generates real-time readmission risk predictions with explainable SHAP outputs
- Tracks inference latency and prediction metadata for observability
- Detects simulated production drift across critical healthcare variables
- Automatically triggers retraining workflows when drift thresholds are exceeded
- Maintains versioned ML model artifacts for lifecycle traceability
- Supports end-to-end ML lifecycle workflows from training to monitoring and retraining


# Deployment

PathwayIQ is deployed as a containerized FastAPI application on AWS EC2.

### Deployment Stack

- Dockerized inference service
- Ubuntu EC2 instance hosting
- FastAPI backend serving predictions
- Public HTTP endpoint for live access
- Serialized XGBoost inference pipeline
- Frontend rendered through FastAPI templates

### Deployment Highlights

- Cross-platform containerized runtime
- Portable filesystem-safe architecture
- Production-style API serving
- Publicly accessible inference endpoint



## Demo

[Watch the Video Demo](https://youtu.be/faQD0bsLIvY)

### Prediction Interface

![Demo Screenshot 1](assets/ss1.png)

### Explainable Risk Output

![Demo Screenshot 2](assets/ss2.png)



## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | Scikit-learn, XGBoost, SHAP |
| Backend | FastAPI, Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Monitoring | Evidently AI, Logging, Model Versioning |
| Deployment | AWS EC2 |



## Machine Learning Pipeline

The ML pipeline includes missing value imputation, categorical encoding, numerical scaling, class imbalance handling, XGBoost classification, and probability calibration through threshold tuning.

The final model was optimized for recall improvement, healthcare interpretability, and deployment realism.


## Model Performance

| Metric | Value |
|---|---|
| Model | XGBoost Classifier |
| ROC-AUC | ~0.69 |
| Recall (Readmission Class) | ~0.60 |
| Precision (Readmission Class) | ~0.19 |
| F1 Score | ~0.28 |
| Monitoring Coverage | 4 critical clinical drift features |
| Retraining Strategy | Threshold-triggered retraining |


## Explainability Layer

PathwayIQ integrates SHAP explainability to provide patient-level explanations, local prediction reasoning, and top contributing risk factors. Example factors include previous inpatient visits, emergency visit frequency, heart failure diagnosis, and discharge disposition patterns. This improves model transparency for healthcare decision support.


## Monitoring and Drift Detection

The system includes simulated production drift monitoring. Drift is intentionally introduced by altering emergency visit frequency, inpatient admissions, hospital stay duration, and diagnosis complexity. Manual threshold-based drift detection triggers retraining workflows.


## Automated Retraining Workflow

When drift exceeds threshold limits:

1. Drift is detected
2. Historical and current data are combined
3. New XGBoost pipeline is retrained
4. Versioned model artifact is generated
5. Retraining event is logged

Generated artifacts:

```
models/
└── versions/
    ├── xgb_model_v1.pkl
    ├── xgb_model_v2.pkl
    └── ...
```

---

## Project Structure

```
pathwayIQ/
│
├── api/
│   └── main.py
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── monitoring/
│   └── drift_report.py
│
├── retraining/
│   └── retrain.py
│
├── models/
│   └── versions/
│
├── logs/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
└── requirements.txt
```

---

## API Reference

### `POST /predict` — Predict Readmission Risk

**Sample Response**

```json
{
  "request_id": "a1b2c3",
  "risk_score": 0.6622,
  "risk_label": "high",
  "latency_ms": 148.2,
  "top_risk_factors": [
    {
      "feature": "Previous inpatient visits",
      "impact": 0.5317
    }
  ]
}
```



## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/Himanshu-2678/pathwayIQ.git
cd pathwayIQ
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the FastAPI server**

```bash
uvicorn api.main:app --reload
```

**5. Open in browser**

```
http://127.0.0.1:8000
```


## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software with proper attribution.

See the [MIT License](LICENSE) file for full details.
