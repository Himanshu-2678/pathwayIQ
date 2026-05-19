LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70


def categorize_risk(probability: float) -> str:

    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk"

    elif probability < HIGH_RISK_THRESHOLD:
        return "Moderate Risk"

    return "High Risk"