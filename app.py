"""ChurnGuard Enterprise — dual-model (RF + Neural Net) churn intelligence."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from churn_dual.cloud import is_streamlit_cloud
from churn_dual.data import load_data
from churn_dual.features import DEMO_PROFILES, LIVE_FEATURE_DEFAULTS, align_features, prepare_churn_df
from churn_dual.model import nn_predict_proba, train_dual

st.set_page_config(page_title="ChurnGuard Enterprise", page_icon="🛡️", layout="wide")

st.markdown("""
<div style="background:linear-gradient(135deg,#7c3aed,#db2777);padding:1.5rem 2rem;border-radius:14px;color:white;">
<h1 style="margin:0;">🛡️ ChurnGuard Enterprise</h1>
<p style="margin:0.4rem 0 0;">Random Forest + Neural Net · Model comparison · Live & batch scoring</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎬 Demo")
    if is_streamlit_cloud():
        st.caption("Cloud: models train on a 4k-row sample for speed (~20s first load).")
    uploaded = st.file_uploader("Training CSV", type=["csv"])
    if st.button("Retrain", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

try:
    df = prepare_churn_df(load_data(uploaded))
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

k1, k2, k3 = st.columns(3)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Churn rate", f"{df['Exited'].mean():.1%}")
k3.metric("Churned", f"{int(df['Exited'].sum()):,}")


@st.cache_resource(show_spinner="Training models (first visit ~15–25s on Cloud)…")
def get_models(_data_hash: str, data: pd.DataFrame):
    del _data_hash
    return train_dual(data, use_cache=True)


data_hash = f"{len(df)}-{df['Exited'].sum()}"
try:
    art = get_models(data_hash, df)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.info("Try **Retrain** in the sidebar, or upload a CSV with an **Exited** column.")
    st.stop()

if not art:
    st.error("Training failed — need at least 50 rows and both churn classes in **Exited**.")
    st.stop()

rf, nn, pre = art["rf"], art["nn"], art["pre"]
Xt = align_features(art["X_test"], art["X_train_cols"])
yt = art["y_test"]
Xtp = pre.transform(Xt)

rf_proba = rf.predict_proba(Xt)[:, 1]
nn_proba = nn_predict_proba(nn, Xtp)
rf_pred, nn_pred = (rf_proba >= 0.5).astype(int), (nn_proba >= 0.5).astype(int)

rf_auc = roc_auc_score(yt, rf_proba)
nn_auc = roc_auc_score(yt, nn_proba)
agreement = (rf_pred == nn_pred).mean()

st.success("Models ready — RF and Neural Net trained on holdout split below.")

cards = [
    {
        "icon": "🎯",
        "title": "Model parity",
        "body": f"RF AUC **{rf_auc:.3f}** · NN AUC **{nn_auc:.3f}** — both rank churn risk similarly.",
        "tone": "positive",
    },
    {
        "icon": "🤝",
        "title": "Agreement",
        "body": f"**{agreement:.0%}** of holdout customers get the same class from both models.",
        "tone": "neutral",
    },
    {
        "icon": "📊",
        "title": "Live demo",
        "body": "Use **Live predict** with preset profiles or **Batch** to export dual scores.",
        "tone": "neutral",
    },
]
html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0.75rem;margin:1rem 0;">'
for c in cards:
    border = {"positive": "#22c55e", "warning": "#f59e0b", "neutral": "#8b5cf6"}.get(c["tone"], "#8b5cf6")
    html += (
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid {border};'
        f'border-radius:10px;padding:0.9rem 1rem;"><small>{c["icon"]} {c["title"]}</small>'
        f'<div style="margin-top:0.35rem;font-size:0.92rem;">{c["body"]}</div></div>'
    )
html += "</div>"
st.markdown(html, unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["📊 Model compare", "🎯 Live predict", "📦 Batch"])

with t1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RF AUC", f"{rf_auc:.3f}")
    m2.metric("NN AUC", f"{nn_auc:.3f}")
    m3.metric("RF F1", f"{f1_score(yt, rf_pred, zero_division=0):.3f}")
    m4.metric("NN F1", f"{f1_score(yt, nn_pred, zero_division=0):.3f}")

    cmp = pd.DataFrame({
        "Model": ["Random Forest", "Neural Net"],
        "AUC": [rf_auc, nn_auc],
        "F1": [f1_score(yt, rf_pred, zero_division=0), f1_score(yt, nn_pred, zero_division=0)],
    })
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(cmp, x="Model", y="AUC", title="Holdout AUC", text_auto=".3f"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(cmp, x="Model", y="F1", title="Holdout F1", text_auto=".3f"), use_container_width=True)

    cm_rf = confusion_matrix(yt, rf_pred)
    cm_nn = confusion_matrix(yt, nn_pred)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(
            px.imshow(cm_rf, text_auto=True, title="RF confusion matrix", color_continuous_scale="Blues"),
            use_container_width=True,
        )
    with cc2:
        st.plotly_chart(
            px.imshow(cm_nn, text_auto=True, title="NN confusion matrix", color_continuous_scale="Purples"),
            use_container_width=True,
        )

    st.caption(
        "Metrics are on a 20% holdout set. F1 uses a 0.5 threshold; "
        "on imbalanced churn data, AUC is often the better headline metric."
    )

with t2:
    profile = st.selectbox("Demo profile", ["Custom"] + list(DEMO_PROFILES.keys()))
    c1, c2 = st.columns(2)
    defaults = DEMO_PROFILES.get(profile, LIVE_FEATURE_DEFAULTS) if profile != "Custom" else LIVE_FEATURE_DEFAULTS
    with c1:
        row = {k: defaults.get(k, LIVE_FEATURE_DEFAULTS.get(k)) for k in LIVE_FEATURE_DEFAULTS}
        row["CreditScore"] = st.slider("Credit", 300, 850, int(row["CreditScore"]))
        _geo = ["France", "Spain", "Germany"]
        row["Geography"] = st.selectbox("Geo", _geo, index=_geo.index(row["Geography"]) if row["Geography"] in _geo else 0)
        row["Gender"] = st.selectbox("Gender", ["Male", "Female"], index=0 if row["Gender"] == "Male" else 1)
        row["Age"] = st.slider("Age", 18, 80, int(row["Age"]))
        row["Tenure"] = st.slider("Tenure", 0, 10, int(row["Tenure"]))
    with c2:
        row["Balance"] = st.number_input("Balance", 0.0, 300000.0, float(row["Balance"]))
        row["EstimatedSalary"] = st.number_input("Salary", 0.0, 200000.0, float(row["EstimatedSalary"]))
    if st.button("Predict churn risk", type="primary", use_container_width=True):
        X = pd.DataFrame([row]).reindex(columns=art["X_train_cols"], fill_value=0)
        rp = float(rf.predict_proba(X)[0, 1])
        np_ = float(nn_predict_proba(nn, pre.transform(X))[0])
        avg = (rp + np_) / 2
        st.metric("Ensemble risk (avg)", f"{avg:.1%}")
        g1, g2 = st.columns(2)
        g1.metric("Random Forest", f"{rp:.1%}")
        g2.metric("Neural Net", f"{np_:.1%}")
        risk_label = "High" if avg >= 0.5 else "Low"
        st.info(f"**{risk_label} churn risk** — use with retention playbooks in your CRM.")

with t3:
    st.write("Upload a CSV with the same columns as the training file (including **Exited** if present).")
    f = st.file_uploader("Batch CSV", type=["csv"])
    if f and st.button("Score batch", type="primary"):
        b = prepare_churn_df(pd.read_csv(f))
        X = b.reindex(columns=art["X_train_cols"], fill_value=0)
        b["RF_Prob"] = rf.predict_proba(X)[:, 1]
        b["NN_Prob"] = nn_predict_proba(nn, pre.transform(X))
        b["Ensemble_Prob"] = (b["RF_Prob"] + b["NN_Prob"]) / 2
        b["RF_Churn"] = (b["RF_Prob"] >= 0.5).astype(int)
        b["NN_Churn"] = (b["NN_Prob"] >= 0.5).astype(int)
        st.dataframe(b.head(50), use_container_width=True, hide_index=True)
        st.download_button("Export all scores", b.to_csv(index=False).encode(), "dual_scores.csv", type="primary")

st.caption("ChurnGuard Enterprise · See DEMO.md")
