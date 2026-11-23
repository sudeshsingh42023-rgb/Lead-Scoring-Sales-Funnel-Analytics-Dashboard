"""
Lead Scoring & Sales Funnel Analytics
- cleans lead data
- funnel / source analysis (charts)
- logistic regression lead-scoring model (score 0-100)
- exports scored leads + metrics for the dashboard
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             classification_report, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ---------- 1. Load (real data if present, else synthetic) ----------
real, synth = Path("data/Leads.csv"), Path("data/Leads_synthetic.csv")
if real.exists():
    df, DATA_SOURCE = pd.read_csv(real), "Kaggle Lead Scoring (X Education)"
elif synth.exists():
    df, DATA_SOURCE = pd.read_csv(synth), "SYNTHETIC (replace with data/Leads.csv)"
else:
    raise SystemExit("Run generate_data.py first (or place Leads.csv in data/).")
print("Data source:", DATA_SOURCE, "| rows:", len(df))

# ---------- 2. Clean ----------
df = df.drop_duplicates(subset="Lead Number").copy()
df = df.replace("Select", np.nan)
df["Lead Source"] = df["Lead Source"].str.strip().str.title().replace({"Google": "Google"})
num_cols = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]
cat_cols = ["Lead Origin", "Lead Source", "Do Not Email", "Last Activity",
            "Specialization", "What is your current occupation", "City"]
cat_cols = [c for c in cat_cols if c in df.columns]
# cap outliers at 99th percentile
for c in num_cols:
    df[c] = df[c].clip(upper=df[c].quantile(0.99))
for c in cat_cols:
    df[c] = df[c].fillna("Unknown")

# ---------- 3. Funnel & source analysis ----------
overall = df["Converted"].mean()
by_source = (df.groupby("Lead Source")["Converted"].agg(["count", "mean"])
             .rename(columns={"count": "leads", "mean": "conv_rate"})
             .sort_values("leads", ascending=False))
by_origin = (df.groupby("Lead Origin")["Converted"].agg(["count", "mean"])
             .rename(columns={"count": "leads", "mean": "conv_rate"}))
by_activity = (df.groupby("Last Activity")["Converted"].agg(["count", "mean"])
               .rename(columns={"count": "leads", "mean": "conv_rate"})
               .sort_values("conv_rate", ascending=False))

# engagement funnel (cumulative stages)
funnel = pd.Series({
    "All leads": len(df),
    "Visited website (visits>0)": int((df["TotalVisits"] > 0).sum()),
    "Engaged (>5 min on site)": int((df["Total Time Spent on Website"] > 300).sum()),
    "Converted": int(df["Converted"].sum()),
})

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(funnel.index[::-1], funnel.values[::-1], color="#2f6fb0")
for i, v in enumerate(funnel.values[::-1]):
    ax.text(v, i, f"  {v:,} ({v/len(df):.0%})", va="center", fontsize=9)
ax.set_title("Sales funnel"); ax.set_xlim(0, len(df) * 1.2)
plt.tight_layout(); plt.savefig(OUT / "funnel.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
s = by_source[by_source.leads >= 30].sort_values("conv_rate")
ax.barh(s.index, s.conv_rate * 100, color="#2a9d8f")
ax.axvline(overall * 100, color="red", ls="--", label=f"Overall {overall:.1%}")
ax.set_xlabel("Conversion rate (%)"); ax.set_title("Conversion rate by lead source")
ax.legend(); plt.tight_layout(); plt.savefig(OUT / "source_conversion.png", dpi=150); plt.close()

# ---------- 4. Model ----------
X, y = df[num_cols + cat_cols], df["Converted"]
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, df.index, test_size=0.25, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
model = Pipeline([("pre", pre),
                  ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
model.fit(X_tr, y_tr)

proba = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)
THRESH = 0.5
pred = (proba >= THRESH).astype(int)
prec, rec = precision_score(y_te, pred), recall_score(y_te, pred)
print(classification_report(y_te, pred, digits=3))
print(f"ROC-AUC={auc:.3f}  precision={prec:.3f}  recall={rec:.3f}")

fpr, tpr, _ = roc_curve(y_te, proba)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot(fpr, tpr, label=f"AUC={auc:.2f}"); ax.plot([0, 1], [0, 1], "k--", lw=.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve"); ax.legend(); plt.tight_layout()
plt.savefig(OUT / "roc.png", dpi=150); plt.close()

# top drivers
names = model.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
drivers = pd.concat([coefs.head(8), coefs.tail(8)])
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(drivers.index, drivers.values, color=["#d62828" if v < 0 else "#2a9d8f" for v in drivers.values])
ax.set_title("Top drivers of conversion (logistic coefficients)")
plt.tight_layout(); plt.savefig(OUT / "drivers.png", dpi=150); plt.close()

# ---------- 5. Score ALL leads 0-100 + priority tiers ----------
df["lead_score"] = (model.predict_proba(df[num_cols + cat_cols])[:, 1] * 100).round(1)
df["priority"] = pd.cut(df["lead_score"], [-1, 40, 70, 101], labels=["Cold", "Warm", "Hot"])
df.sort_values("lead_score", ascending=False).to_csv(OUT / "scored_leads.csv", index=False)

# ---------- 6. Business impact (top-30% outreach vs random) ----------
test = pd.DataFrame({"y": y_te.values, "p": proba}).sort_values("p", ascending=False)
top = test.head(int(len(test) * 0.30))
capture = top["y"].sum() / test["y"].sum()
lift = top["y"].mean() / test["y"].mean()
print(f"Calling top 30% of leads captures {capture:.0%} of conversions (lift {lift:.2f}x)")

metrics = {
    "data_source": DATA_SOURCE, "n_leads": int(len(df)),
    "overall_conversion": round(float(overall), 4),
    "roc_auc": round(float(auc), 3), "precision": round(float(prec), 3),
    "recall": round(float(rec), 3),
    "top30_capture": round(float(capture), 3), "top30_lift": round(float(lift), 2),
    "tier_counts": df["priority"].value_counts().to_dict(),
}
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
by_source.to_csv(OUT / "by_source.csv"); by_origin.to_csv(OUT / "by_origin.csv")
by_activity.to_csv(OUT / "by_activity.csv"); funnel.to_csv(OUT / "funnel.csv", header=["count"])
print(json.dumps(metrics, indent=2))
