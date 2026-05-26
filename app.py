"""ChurnGuard Enterprise — dual-model (RF + Neural Net) churn intelligence."""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from churn_dual.data import load_data
from churn_dual.model import train_dual

st.set_page_config(page_title="ChurnGuard Enterprise", page_icon="🛡️", layout="wide")
st.markdown("""
<div style="background:linear-gradient(135deg,#7c3aed,#db2777);padding:1.5rem 2rem;border-radius:14px;color:white;">
<h1 style="margin:0;">🛡️ ChurnGuard Enterprise</h1>
<p style="margin:0.4rem 0 0;">Random Forest + Deep Learning · Model comparison · Live & batch scoring</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Training CSV", type=["csv"])
if st.sidebar.button("Retrain"):
    st.cache_resource.clear()
    st.session_state.pop("art", None)

df = load_data(uploaded)
st.metric("Customers", len(df))
st.metric("Churn rate", f"{df['Exited'].mean():.1%}")

@st.cache_resource
def get_models(data):
    return train_dual(data)

art = get_models(df)
if not art:
    st.error("Training failed")
    st.stop()

rf, nn, pre = art["rf"], art["nn"], art["pre"]
Xt, yt = art["X_test"], art["y_test"]
Xtp = pre.transform(Xt)

rf_proba = rf.predict_proba(Xt)[:, 1]
nn_proba = nn.predict(Xtp).ravel()
rf_pred, nn_pred = (rf_proba >= 0.5).astype(int), (nn_proba >= 0.5).astype(int)

t1, t2, t3 = st.tabs(["📊 Model compare", "🎯 Live predict", "📦 Batch"])

with t1:
    m1, m2 = st.columns(2)
    m1.metric("RF AUC", f"{roc_auc_score(yt, rf_proba):.3f}")
    m1.metric("RF F1", f"{f1_score(yt, rf_pred, zero_division=0):.3f}")
    m2.metric("NN AUC", f"{roc_auc_score(yt, nn_proba):.3f}")
    m2.metric("NN F1", f"{f1_score(yt, nn_pred, zero_division=0):.3f}")
    cmp = pd.DataFrame({"Model": ["Random Forest", "Neural Net"], "AUC": [roc_auc_score(yt, rf_proba), roc_auc_score(yt, nn_proba)]})
    st.plotly_chart(px.bar(cmp, x="Model", y="AUC", title="Holdout AUC comparison"), use_container_width=True)

with t2:
    c1, c2 = st.columns(2)
    with c1:
        row = {
            "CreditScore": st.slider("Credit", 300, 850, 650),
            "Geography": st.selectbox("Geo", ["France", "Spain", "Germany"]),
            "Gender": st.selectbox("Gender", ["Male", "Female"]),
            "Age": st.slider("Age", 18, 80, 40),
            "Tenure": st.slider("Tenure", 0, 10, 3),
        }
    with c2:
        row.update({
            "Balance": st.number_input("Balance", 0.0, 300000.0, 50000.0),
            "EstimatedSalary": st.number_input("Salary", 0.0, 200000.0, 75000.0),
        })
    if st.button("Predict", type="primary"):
        X = pd.DataFrame([row]).reindex(columns=art["X_train_cols"], fill_value=0)
        rp = rf.predict_proba(X)[0, 1]
        np_ = nn.predict(pre.transform(X)).ravel()[0]
        st.success(f"RF churn risk: **{rp:.1%}** · NN churn risk: **{np_:.1%}**")

with t3:
    f = st.file_uploader("Batch CSV", type=["csv"])
    if f and st.button("Score batch"):
        b = pd.read_csv(f)
        X = b.reindex(columns=art["X_train_cols"], fill_value=0)
        b["RF_Prob"] = rf.predict_proba(X)[:, 1]
        b["NN_Prob"] = nn.predict(pre.transform(X)).ravel()
        st.dataframe(b, use_container_width=True)
        st.download_button("Export", b.to_csv(index=False).encode(), "dual_scores.csv")

st.caption("ChurnGuard Enterprise · See DEMO.md")
