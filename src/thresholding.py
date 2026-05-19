def assign_risk_band(prob):
    """
    Clinical risk stratification policy using calibrated probabilities.
    """

    if prob < 0.15:
        return "Low Risk"

    elif prob < 0.30:
        return "Moderate Risk"

    return "High Risk"