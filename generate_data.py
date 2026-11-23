"""
Generates a SYNTHETIC lead dataset that mimics the schema of the public Kaggle
"Lead Scoring" (X Education) dataset. Used only so the project runs end-to-end
without a download.

To use the REAL data: download Leads.csv from Kaggle (search "Lead Scoring Dataset",
X Education), save it as data/Leads.csv and run lead_scoring.py -- it auto-detects it.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 9000

lead_origin = rng.choice(
    ["Landing Page Submission", "API", "Lead Add Form", "Lead Import"],
    N, p=[0.53, 0.39, 0.07, 0.01])
lead_source = rng.choice(
    ["Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference",
     "Welingak Website", "Referral Sites", "Facebook"],
    N, p=[0.31, 0.28, 0.19, 0.12, 0.05, 0.02, 0.02, 0.01])
do_not_email = rng.choice(["Yes", "No"], N, p=[0.08, 0.92])
total_visits = np.clip(rng.poisson(3.4, N), 0, 30)
time_on_site = np.clip(rng.gamma(1.2, 450, N).astype(int), 0, 2200)
page_views = np.clip(rng.poisson(2.4, N), 0, 12)
last_activity = rng.choice(
    ["Email Opened", "SMS Sent", "Olark Chat Conversation", "Page Visited on Website",
     "Converted to Lead", "Email Bounced", "Unsubscribed"],
    N, p=[0.38, 0.30, 0.10, 0.07, 0.05, 0.05, 0.05])
specialization = rng.choice(
    ["Finance", "Marketing", "HR", "Operations", "IT", "Not Specified"],
    N, p=[0.12, 0.11, 0.11, 0.10, 0.05, 0.51])
occupation = rng.choice(
    ["Unemployed", "Working Professional", "Student", "Other"],
    N, p=[0.60, 0.08, 0.03, 0.29])
city = rng.choice(["Mumbai", "Thane", "Other Cities", "Hyderabad", "Bengaluru"],
                  N, p=[0.35, 0.08, 0.30, 0.12, 0.15])

# Hidden "true" conversion logic (so a model can genuinely learn something)
z = (-2.0
     + 0.0017 * time_on_site
     + 0.05 * total_visits
     + 1.1 * (occupation == "Working Professional")
     + 2.4 * (lead_source == "Reference")
     + 2.0 * (lead_source == "Welingak Website")
     + 1.4 * (last_activity == "SMS Sent")
     + 0.5 * (last_activity == "Email Opened")
     - 2.0 * (last_activity == "Email Bounced")
     - 1.8 * (last_activity == "Unsubscribed")
     - 1.5 * (do_not_email == "Yes")
     - 0.9 * (lead_origin == "API")
     + 0.8 * (lead_origin == "Lead Add Form")
     + rng.normal(0, 0.6, N))
p = 1 / (1 + np.exp(-z))
converted = (rng.random(N) < p).astype(int)

df = pd.DataFrame({
    "Lead Number": np.arange(600000, 600000 + N),
    "Lead Origin": lead_origin, "Lead Source": lead_source,
    "Do Not Email": do_not_email, "Converted": converted,
    "TotalVisits": total_visits.astype(float),
    "Total Time Spent on Website": time_on_site,
    "Page Views Per Visit": page_views.astype(float),
    "Last Activity": last_activity, "Specialization": specialization,
    "What is your current occupation": occupation, "City": city,
})
# inject realistic mess: missing values + a "Select" placeholder + messy casing
for col, frac in [("TotalVisits", 0.015), ("Page Views Per Visit", 0.015),
                  ("Lead Source", 0.004), ("City", 0.15)]:
    df.loc[rng.random(N) < frac, col] = np.nan
df.loc[rng.random(N) < 0.2, "Specialization"] = "Select"
df.loc[df["Lead Source"] == "Google", "Lead Source"] = rng.choice(
    ["Google", "google"], (df["Lead Source"] == "Google").sum(), p=[0.97, 0.03])

Path("data").mkdir(exist_ok=True)
df.to_csv("data/Leads_synthetic.csv", index=False)
print(f"Wrote data/Leads_synthetic.csv  rows={len(df)}  conversion={df.Converted.mean():.1%}")
