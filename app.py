import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    model = joblib.load(BASE / "fraud_rf_model.joblib")
    cfg = json.loads((BASE / "fraud_rf_config.json").read_text())
    return model, cfg


@st.cache_data
def load_demo():
    return pd.read_csv(BASE / "demo_test_transactions.csv")


def prepare(df, features):
    """Add engineered columns if the file has raw Time/Amount."""
    df = df.copy()
    if "Hour" not in df.columns and "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    if "Amount_log" not in df.columns and "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])
    missing = [c for c in features if c not in df.columns]
    return df, missing


model, cfg = load_model()
features = cfg["features"]

st.title("Credit Card Fraud Detection")
st.caption(f"Model: {cfg['model']} | Tuned decision threshold: {cfg['threshold']:.3f}")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
source = st.sidebar.radio("Data source", ["Demo transactions", "Upload CSV"])
threshold = st.sidebar.slider(
    "Fraud threshold", 0.01, 0.99, float(round(cfg["threshold"], 2)), 0.01,
    help="A transaction is flagged as fraud when its fraud probability is at or above this value. "
         "Lower = catch more fraud but more false alarms.",
)

if source == "Demo transactions":
    demo = load_demo()
    n_legit = st.sidebar.slider("Legitimate transactions", 100, 5000, 1000, 100)
    n_fraud = st.sidebar.slider("Fraud transactions", 0, 50, 10)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
    if "Class" in demo.columns:
        fraud = demo[demo["Class"] == 1]
        legit = demo[demo["Class"] == 0]
        data = pd.concat([
            legit.sample(min(n_legit, len(legit)), random_state=int(seed)),
            fraud.sample(min(n_fraud, len(fraud)), random_state=int(seed)),
        ]).sample(frac=1, random_state=int(seed)).reset_index(drop=True)
    else:
        data = demo.sample(min(n_legit, len(demo)), random_state=int(seed)).reset_index(drop=True)
else:
    up = st.sidebar.file_uploader("CSV with V1-V28 plus Time/Amount (or Hour/Amount_log)", type="csv")
    if up is None:
        st.info("Upload a CSV to score it, or switch to the demo transactions.")
        st.stop()
    data = pd.read_csv(up)

data, missing = prepare(data, features)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------------- Scoring ----------------
data["fraud_probability"] = model.predict_proba(data[features])[:, 1]
data["flagged"] = (data["fraud_probability"] >= threshold).astype(int)

c1, c2, c3 = st.columns(3)
c1.metric("Transactions scored", f"{len(data):,}")
c2.metric("Flagged as fraud", f"{int(data['flagged'].sum()):,}")
c3.metric("Flag rate", f"{data['flagged'].mean():.2%}")

# ---------------- Evaluation (if labels present) ----------------
if "Class" in data.columns and data["Class"].nunique() == 2:
    y, p, s = data["Class"], data["flagged"], data["fraud_probability"]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    st.subheader("Performance on this sample")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{precision_score(y, p, zero_division=0):.3f}")
    m2.metric("Recall", f"{recall_score(y, p):.3f}")
    m3.metric("F1", f"{f1_score(y, p):.3f}")
    m4.metric("ROC-AUC", f"{roc_auc_score(y, s):.3f}")
    m5.metric("PR-AUC", f"{average_precision_score(y, s):.3f}")

    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual: Legitimate", "Actual: Fraud"],
        columns=["Predicted: Legitimate", "Predicted: Fraud"],
    )
    st.dataframe(cm)
    st.caption(
        "This sample over-represents fraud compared with real life (about 0.17%), "
        "so precision here looks better than it would in production."
    )

# ---------------- Transactions table ----------------
st.subheader("Transactions")
show_flagged = st.checkbox("Show only flagged transactions", value=True)
cols = ["fraud_probability", "flagged"]
if "Class" in data.columns:
    cols.append("Class")
if "Amount" in data.columns:
    cols.append("Amount")
if "Time" in data.columns:
    cols.append("Time")
cols += [c for c in ["Hour", "Amount_log", "V1", "V2", "V3", "V4"] if c in data.columns and c not in cols]

view = data[data["flagged"] == 1] if show_flagged else data
view = view.sort_values("fraud_probability", ascending=False)
st.dataframe(view[cols].head(500), use_container_width=True)

st.download_button(
    "Download all scored transactions (CSV)",
    data.to_csv(index=False).encode("utf-8"),
    file_name="scored_transactions.csv",
    mime="text/csv",
)
