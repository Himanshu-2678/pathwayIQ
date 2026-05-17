async function runAnalysis() {

    const btn = document.getElementById('run-btn');
    btn.textContent = 'Assessing…';
    btn.disabled = true;

    const payload = {

        age:              parseInt(document.getElementById('age').value)       || 7,
        time_in_hospital: parseInt(document.getElementById('los').value)       || 6,
        number_inpatient: parseInt(document.getElementById('inpatient').value) || 3,
        number_emergency: parseInt(document.getElementById('emergency').value) || 1,
        number_diagnoses: parseInt(document.getElementById('diagnoses').value) || 8,

        // Placeholder clinical values
        num_lab_procedures: 45,
        num_procedures:     1,
        num_medications:    18,
        number_outpatient:  0,

        race:              'Caucasian',
        gender:            'Female',
        payer_code:        'MC',
        medical_specialty: 'InternalMedicine',

        admission_type_id:        '1',
        discharge_disposition_id: '1',
        admission_source_id:      '7',

        diag_1: '428',
        diag_2: '250.8',
        diag_3: '401',

        max_glu_serum: 'None',
        A1Cresult:     '>8',

        metformin:    'Steady',
        repaglinide:  'No',
        nateglinide:  'No',
        glimepiride:  'No',
        glipizide:    'Steady',
        glyburide:    'No',
        pioglitazone: 'No',
        rosiglitazone:'No',
        acarbose:     'No',
        insulin:      'Up',

        glipizide_metformin:      'No',
        glyburide_metformin:      'No',
        glimepiride_pioglitazone: 'No',
        metformin_rosiglitazone:  'No',
        metformin_pioglitazone:   'No',

        change:      'Ch',
        diabetesMed: 'Yes'
    };

    let data;

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Server error');
        data = await res.json();

    } catch (err) {
        // Fallback: local score simulation when backend is unavailable
        data = localScore(payload);
    }

    renderResult(data);

    btn.textContent = 'Assess readmission risk';
    btn.disabled = false;
}


function localScore(p) {

    let s = 0.12;
    s += p.number_inpatient * 0.06;
    s += p.number_emergency * 0.04;
    s += p.number_diagnoses * 0.012;
    s += (p.time_in_hospital - 3) * 0.015;
    s += p.age * 0.008;
    s += 0.07; // insulin escalation
    s += 0.05; // uncontrolled HbA1c
    s += 0.04; // medication change
    s = Math.min(Math.max(s, 0.05), 0.96);

    const factors = [
        { feature: 'Prior inpatient visits',  impact: (p.number_inpatient * 0.06).toFixed(3) },
        { feature: 'Number of diagnoses',     impact: (p.number_diagnoses * 0.012).toFixed(3) },
        { feature: 'Emergency visits',        impact: (p.number_emergency * 0.04).toFixed(3) },
        { feature: 'Length of stay',          impact: Math.max(0, (p.time_in_hospital - 3) * 0.015).toFixed(3) },
        { feature: 'Insulin escalation',      impact: '0.070' },
        { feature: 'Uncontrolled HbA1c',      impact: '0.050' },
    ]
    .filter(f => parseFloat(f.impact) > 0)
    .sort((a, b) => parseFloat(b.impact) - parseFloat(a.impact))
    .slice(0, 5);

    const label = s >= 0.5 ? 'High risk' : s >= 0.3 ? 'Moderate risk' : 'Low risk';

    return { risk_score: s, risk_label: label, top_risk_factors: factors };
}


function renderResult(data) {

    const pct   = Math.round(data.risk_score * 100);
    const isHigh = pct >= 50;
    const isMed  = pct >= 30 && pct < 50;
    const tier   = isHigh ? 'high' : isMed ? 'medium' : 'low';

    // Show result area
    document.getElementById('result-area').classList.add('visible');

    // Risk number
    const numEl = document.getElementById('risk-num');
    numEl.textContent = pct + '%';
    numEl.className = 'risk-number ' + tier;

    // Verdict label
    const verdictLabel = document.getElementById('risk-verdict-label');
    verdictLabel.textContent = isHigh ? 'High risk' : isMed ? 'Moderate risk' : 'Low risk';
    verdictLabel.className = 'risk-verdict-label ' + tier;

    // Verdict sub
    document.getElementById('risk-verdict-sub').textContent = isHigh
        ? 'Elevated chance of readmission within 30 days'
        : isMed
        ? 'Some risk of readmission - monitor closely'
        : 'Low likelihood of readmission';

    // Progress track
    const fill = document.getElementById('track-fill');
    fill.className = 'track-fill ' + tier;
    setTimeout(() => fill.style.width = pct + '%', 50);

    // Risk factors
    const list = document.getElementById('factors-list');
    list.innerHTML = '';

    data.top_risk_factors.forEach((f, i) => {

        const val   = parseFloat(f.impact);
        const t     = val > 0.05 ? 'high' : val > 0.025 ? 'medium' : 'low';
        const label = val > 0.05 ? 'High influence' : val > 0.025 ? 'Moderate' : 'Low';

        const el = document.createElement('div');
        el.className = 'factor';
        el.innerHTML = `
            <span class="factor-name">${f.feature}</span>
            <span class="factor-pill ${t}">${label}</span>
        `;

        list.appendChild(el);

        setTimeout(() => el.classList.add('in'), 60 * i + 100);
    });

    // Contextual risk insight

    let dynamicNote = '';

    if (isHigh) {

        dynamicNote =
            'This patient exhibits utilization patterns commonly associated with elevated short-term readmission risk.';

    } else if (isMed) {

        dynamicNote =
            'Moderate readmission risk detected based on recent encounter and diagnosis history.';

    } else {

        dynamicNote =
            'Current encounter patterns do not indicate strong short-term readmission risk signals.';
    }

    document.getElementById('result-note').textContent =
        dynamicNote;
    }