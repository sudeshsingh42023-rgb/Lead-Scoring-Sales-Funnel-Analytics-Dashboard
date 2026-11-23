# Lead Scoring & Sales Funnel Analytics Dashboard

Cleans lead data, analyses the funnel and lead sources, trains a logistic regression lead-scoring model (score 0-100, Hot/Warm/Cold), and serves a Streamlit dashboard with a downloadable call list.

> **Data note:** `generate_data.py` creates SYNTHETIC data in the schema of the Kaggle "Lead Scoring" (X Education) dataset so the project runs without a download. For real results, download `Leads.csv` from Kaggle, save it as `data/Leads.csv`, and re-run; `lead_scoring.py` picks it up automatically. Update any numbers you quote on your resume.

## Run
```bash
pip install -r requirements.txt
python generate_data.py      # skip if you use the real Leads.csv
python lead_scoring.py
streamlit run app.py
```

## Pipeline
1. **Clean:** dedupe, treat "Select" as missing, normalise text, cap outliers at the 99th percentile, impute
2. **Analyse:** funnel stages, conversion by source / origin / last activity
3. **Model:** one-hot + scaling pipeline, class-balanced logistic regression, stratified 75/25 split; ROC-AUC, precision, recall
4. **Score:** every lead gets a 0-100 score and a Hot / Warm / Cold tier
5. **Impact:** share of conversions captured by calling only the top 30% of leads, and lift vs random calling

## Outputs (`outputs/`)
`scored_leads.csv`, `metrics.json`, `funnel.png`, `source_conversion.png`, `roc.png`, `drivers.png`

## Limitations
- Logistic regression chosen for interpretability (coefficients explain drivers to sales teams); gradient boosting could raise AUC at some cost to explainability.
- Leakage check: drop any feature recorded after the sale outcome (for example "Tags" in the Kaggle data).
