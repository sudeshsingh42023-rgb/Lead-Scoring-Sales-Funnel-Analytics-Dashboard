"""Streamlit dashboard.  Run:  streamlit run app.py   (after lead_scoring.py)"""
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lead Scoring Dashboard", layout="wide")
st.title("Lead Scoring & Sales Funnel Dashboard")

m = json.load(open("outputs/metrics.json"))
df = pd.read_csv("outputs/scored_leads.csv")
st.caption(f"Data: {m['data_source']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Leads", f"{m['n_leads']:,}")
c2.metric("Overall conversion", f"{m['overall_conversion']:.1%}")
c3.metric("Model ROC-AUC", m["roc_auc"])
c4.metric("Top-30% lift", f"{m['top30_lift']}x")

left, right = st.columns(2)
left.image("outputs/funnel.png", caption="Funnel")
right.image("outputs/source_conversion.png", caption="Conversion by source")

st.subheader("Call list: filter and download")
f1, f2, f3 = st.columns(3)
tiers = f1.multiselect("Priority", ["Hot", "Warm", "Cold"], default=["Hot"])
sources = f2.multiselect("Lead source", sorted(df["Lead Source"].unique()))
min_score = f3.slider("Min score", 0, 100, 70)

view = df[df["priority"].isin(tiers) & (df["lead_score"] >= min_score)]
if sources:
    view = view[view["Lead Source"].isin(sources)]
st.write(f"{len(view):,} leads match")
cols = ["Lead Number", "lead_score", "priority", "Lead Source", "Lead Origin",
        "Last Activity", "Total Time Spent on Website"]
st.dataframe(view[cols].head(500), use_container_width=True)
st.download_button("Download call list (CSV)", view[cols].to_csv(index=False),
                   "call_list.csv", "text/csv")

st.subheader("Model drivers")
st.image("outputs/drivers.png")
