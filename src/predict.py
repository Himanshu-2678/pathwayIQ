FEATURE_NAME_MAP = {

    "num__number_inpatient": "Previous inpatient visits",
    "num__number_emergency": "Emergency visit frequency",
    "num__number_diagnoses": "Diagnosis complexity",
    "num__time_in_hospital": "Length of hospital stay",
    "num__num_medications": "Medication count",
    "num__num_lab_procedures": "Lab procedures count",
    "num__age": "Patient age",

    "cat__diag_1_428": "Primary diagnosis: heart failure",
    "cat__diag_1_486": "Primary diagnosis: pneumonia",

    "cat__discharge_disposition_id_1": "Discharge disposition pattern",

    "cat__payer_code_Unknown": "Missing payer information",

    "cat__insulin_Down": "Reduced insulin dosage",

    "cat__diabetesMed_No": "No diabetes medication"
}

def clean_feature_name(feature_name):

    return FEATURE_NAME_MAP.get(
        feature_name,
        feature_name.replace("num__", "")
                    .replace("cat__", "")
                    .replace("_", " ")
    )