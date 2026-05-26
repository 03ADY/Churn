"""ChurnGuard Enterprise — dual-model (RF + Neural Net) churn intelligence."""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score, roc_auc_score

from churn_dual.data import load_data
from churn_dual.features import LIVE_FEATURE_DEFAULTS, prepare_churn_df
from churn_dual.model import nn_predict_proba, train_dual

st.set_page_config(page_title="ChurnGuard Enterprise", page_icon="🛡️", layout="wide")
st.markdown("""
<div style="background:linear-gradient(135deg,#7c3aed,#db2777);padding:1.5rem 2rem;border-radius:14px;color:white;">
<h1 style="margin:0;">🛡️ ChurnGuard Enterprise</h1>
<p style="margin:0.4rem 0 0;">Random Forest + Neural Net · Model comparison · Live & batch scoring</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Training CSV", type=["csv"])
if st.sidebar.button("Retrain"):
    st.cache_resource.clear()
    st.session_state.pop("art", None)

try:
    df = prepare_churn_df(load_data(uploaded))
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

st.metric("Customers", len(df))
st.metric("Churn rate", f"{df['Exited'].mean():.1%}")


@st.cache_resource(show_spinner="Training models (first visit may take ~30s)…")
def get_models(_data_hash: str, data: pd.DataFrame):
    del _data_hash  # cache key only
    return train_dual(data)


data_hash = f"{len(df)}-{df['Exited'].sum()}"
try:
    with st.spinner("Loading models…"):
        art = get_models(data_hash, df)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.info("Try uploading a CSV with an **Exited** column, or use **Retrain** after fixing data.")
    st.stop()

if not art:
    st.error("Training failed — need at least 50 rows and both churn classes in **Exited**.")
    st.stop()

rf, nn, pre = art["rf"], art["nn"], art["pre"]
Xt, yt = art["X_test"], art["y_test"]
Xtp = pre.transform(Xt)

rf_proba = rf.predict_proba(Xt)[:, 1]
nn_proba = nn_predict_proba(nn, Xtp)
rf_pred, nn_pred = (rf_proba >= 0.5).astype(int), (nn_proba >= 0.5).astype(int)

t1, t2, t3 = st.tabs(["📊 Model compare", "🎯 Live predict", "📦 Batch"])

with t1:
    m1, m2 = st.columns(2)
    m1.metric("RF AUC", f"{roc_auc_score(yt, rf_proba):.3f}")
    m1.metric("RF F1", f"{f1_score(yt, rf_pred, zero_division=0):.3f}")
    m2.metric("NN AUC", f"{roc_auc_score(yt, nn_proba):.3f}")
    m2.metric("NN F1", f"{f1_score(yt, nn_pred, zero_division=0):.3f}")
    cmp = pd.DataFrame({
        "Model": ["Random Forest", "Neural Net"],
        "AUC": [roc_auc_score(yt, rf_proba), roc_auc_score(yt, nn_proba)],
    })
    st.plotly_chart(px.bar(cmp, x="Model", y="AUC", title="Holdout AUC comparison"), use_container_width=True)

with t2:
    c1, c2 = st.columns(2)
    with c1:
        row = {k: LIVE_FEATURE_DEFAULTS[k] for k in LIVE_FEATURE_DEFAULTS}
        row["CreditScore"] = st.slider("Credit", 300, 850, 650)
        row["Geography"] = st.selectbox("Geo", ["France", "Spain", "Germany"])
        row["Gender"] = st.selectbox("Gender", ["Male", "Female"])
        row["Age"] = st.slider("Age", 18, 80, 40)
        row["Tenure"] = st.slider("Tenure", 0, 10, 3)
    with c2:
        row["Balance"] = st.number_input("Balance", 0.0, 300000.0, 50000.0)
        row["EstimatedSalary"] = st.number_input("Salary", 0.0, 200000.0, 75000.0)
        if "NumOfProducts" in art["X_train_cols"]:
            row["NumOfProducts"] = st.slider("Products", 1, 4, 1)
        if "HasCrCard" in art["X_train_cols"]:
            row["HasCrCard"] = st.selectbox("Has credit card", [1, 0])
        if "IsActiveMember" in art["X_train_cols"]:
            row["IsActiveMember"] = st.selectbox("Active member", [1, 0])
    if st.button("Predict", type="primary"):
        X = pd.DataFrame([row]).reindex(columns=art["X_train_cols"], fill_value=0)
        rp = float(rf.predict_proba(X)[0, 1])
        np_ = float(nn_predict_proba(nn, pre.transform(X))[0])
        st.success(f"RF churn risk: **{rp:.1%}** · NN churn risk: **{np_:.1%}**")

with t3:
    f = st.file_uploader("Batch CSV", type=["csv"])
    if f and st.button("Score batch"):
        b = prepare_churn_df(pd.read_csv(f))
        X = b.reindex(columns=art["X_train_cols"], fill_value=0)
        b["RF_Prob"] = rf.predict_proba(X)[:, 1]
        b["NN_Prob"] = nn_predict_proba(nn, pre.transform(X))
        st.dataframe(b, use_container_width=True)
        st.download_button("Export", b.to_csv(index=False).encode(), "dual_scores.csv")

st.caption("ChurnGuard Enterprise · See DEMO.md")
