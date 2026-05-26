"""ChurnGuard Enterprise — dual-model (RF + Neural Net) churn intelligence."""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from churn_dual.cloud import is_streamlit_cloud
from churn_dual.data import load_data
from churn_dual.explain import (
    batch_export_help,
    compare_models,
    confusion_compare_note,
    confusion_narrative,
    glossary_markdown,
    holdout_metric_cards,
    interpret_agreement,
    interpret_churn_rate,
    interpret_live_risk,
)
from churn_dual.features import DEMO_PROFILES, LIVE_FEATURE_DEFAULTS, align_features, prepare_churn_df
from churn_dual.model import nn_predict_proba, train_dual
from churn_dual.theme import hero_html, inject_theme, style_fig

st.set_page_config(page_title="ChurnGuard Enterprise", page_icon="🛡️", layout="wide")
inject_theme()
st.markdown(
    hero_html(
        "ChurnGuard Enterprise",
        "Random Forest + Neural Net · Explained metrics · Live & batch scoring",
        "🛡️",
    ),
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎬 Demo")
    if is_streamlit_cloud():
        st.caption("Cloud: models train on a 4k-row sample for speed (~20s first load).")
    uploaded = st.file_uploader("Training CSV", type=["csv"])
    if st.button("Retrain", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    with st.expander("📖 Metric glossary"):
        st.markdown(glossary_markdown())

try:
    df = prepare_churn_df(load_data(uploaded))
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

churn_rate = float(df["Exited"].mean())
k1, k2, k3 = st.columns(3)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Churn rate", f"{churn_rate:.1%}")
k3.metric("Churned", f"{int(df['Exited'].sum()):,}")
st.markdown(interpret_churn_rate(churn_rate))


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
rf_f1 = f1_score(yt, rf_pred, zero_division=0)
nn_f1 = f1_score(yt, nn_pred, zero_division=0)
agreement = (rf_pred == nn_pred).mean()

cards = holdout_metric_cards(rf_auc, nn_auc, rf_f1, nn_f1)

st.markdown("### Executive summary")
st.markdown(
    compare_models(rf_auc, nn_auc)
    + " "
    + interpret_agreement(agreement)
)
if cards["auc_combined"]:
    st.info(cards["auc_summary"])
else:
    c1, c2 = st.columns(2)
    with c1:
        st.info(cards["rf_auc_x"]["summary"])
    with c2:
        st.info(cards["nn_auc_x"]["summary"])
if cards["f1_combined"]:
    st.caption(cards["f1_summary"])
else:
    f1a, f1b = st.columns(2)
    with f1a:
        st.caption(cards["rf_f1_x"]["summary"])
    with f1b:
        st.caption(cards["nn_f1_x"]["summary"])

t1, t2, t3 = st.tabs(["📊 Model compare", "🎯 Live predict", "📦 Batch"])

with t1:
    st.markdown(
        "We hold back **20% of customers** the models never saw during training, then measure "
        "how well each model predicts who actually churned (`Exited = 1`)."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RF AUC", f"{rf_auc:.3f}", help="Random Forest ranking quality")
    m2.metric("NN AUC", f"{nn_auc:.3f}", help="Neural Net ranking quality")
    m3.metric("RF F1", f"{rf_f1:.3f}", help="RF at 50% risk cutoff")
    m4.metric("NN F1", f"{nn_f1:.3f}", help="NN at 50% risk cutoff")

    with st.expander("What do these numbers mean?", expanded=True):
        if cards["auc_combined"]:
            st.markdown(cards["auc_detail_once"])
        else:
            st.markdown(cards["rf_auc_x"]["detail"])
            st.markdown(
                f"**Random Forest** AUC {rf_auc:.3f}: {cards['rf_auc_x']['summary']}  \n"
                f"**Neural Net** AUC {nn_auc:.3f}: {cards['nn_auc_x']['summary']}"
            )
        if cards["f1_combined"]:
            st.markdown(cards["f1_detail_once"])
        else:
            st.markdown(cards["rf_f1_x"]["detail"])
            st.markdown(
                f"**RF F1** {rf_f1:.3f}: {cards['rf_f1_x']['summary']}  \n"
                f"**NN F1** {nn_f1:.3f}: {cards['nn_f1_x']['summary']}"
            )
        st.markdown(f"**Model agreement:** {interpret_agreement(agreement)}")

    cmp = pd.DataFrame({
        "Model": ["Random Forest", "Neural Net"],
        "AUC": [rf_auc, nn_auc],
        "F1": [rf_f1, nn_f1],
    })
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(style_fig(px.bar(cmp, x="Model", y="AUC", title="Holdout AUC (higher = better ranking)", text_auto=".3f")), use_container_width=True)
    with ch2:
        st.plotly_chart(style_fig(px.bar(cmp, x="Model", y="F1", title="Holdout F1 (threshold 50%)", text_auto=".3f")), use_container_width=True)

    cm_rf = confusion_matrix(yt, rf_pred)
    cm_nn = confusion_matrix(yt, nn_pred)
    note = confusion_compare_note(rf_auc, nn_auc, rf_f1, nn_f1)
    if note:
        st.markdown(note)
    st.markdown(confusion_narrative(cm_rf, "Random Forest"))
    st.markdown(confusion_narrative(cm_nn, "Neural Net"))

    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(style_fig(px.imshow(cm_rf, text_auto=True, title="RF confusion matrix", color_continuous_scale="Blues")), use_container_width=True)
        st.caption("Rows/columns: Predicted Stay/Churn vs Actual Stay/Churn")
    with cc2:
        st.plotly_chart(style_fig(px.imshow(cm_nn, text_auto=True, title="NN confusion matrix", color_continuous_scale="Purples")), use_container_width=True)

    st.markdown("#### Recommended use")
    st.markdown(
        "- **Prioritize outreach** using **AUC** and **Ensemble_Prob** (top 10–20% risk), not only the 50% yes/no flags.  \n"
        "- **Random Forest** for explainability in meetings; **Neural Net** as a corroborating score.  \n"
        "- Re-train when you upload a new CSV or click **Retrain** after material business changes."
    )

with t2:
    st.markdown(
        "Enter a customer profile (or pick a demo). The models output **probability of churn** — "
        "the chance they leave in a similar way to past churners in your data."
    )
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
        live = interpret_live_risk(rp, np_)
        st.metric("Ensemble risk (avg)", f"{live['avg']:.1%}")
        g1, g2 = st.columns(2)
        g1.metric("Random Forest says", f"{rp:.1%}", help="Chance of churn according to RF")
        g2.metric("Neural Net says", f"{np_:.1%}", help="Chance of churn according to NN")
        st.success(live["summary"])
        with st.expander("Full explanation"):
            st.markdown(live["detail"])
            st.markdown(f"**Suggested action:** {live['playbook']}")

with t3:
    st.markdown(
        "Score many customers at once. Export the file and sort by **Ensemble_Prob** "
        "to build your retention call list."
    )
    st.markdown(batch_export_help())
    f = st.file_uploader("Batch CSV", type=["csv"])
    if f and st.button("Score batch", type="primary"):
        b = prepare_churn_df(pd.read_csv(f))
        X = b.reindex(columns=art["X_train_cols"], fill_value=0)
        b["RF_Prob"] = rf.predict_proba(X)[:, 1]
        b["NN_Prob"] = nn_predict_proba(nn, pre.transform(X))
        b["Ensemble_Prob"] = (b["RF_Prob"] + b["NN_Prob"]) / 2
        b["RF_Churn"] = (b["RF_Prob"] >= 0.5).astype(int)
        b["NN_Churn"] = (b["NN_Prob"] >= 0.5).astype(int)
        high = int((b["Ensemble_Prob"] >= 0.5).sum())
        st.markdown(
            f"**{high:,} customers ({high/len(b):.1%})** scored at or above 50% ensemble risk — "
            f"candidates for retention; consider contacting the **top 10%** first for best ROI."
        )
        st.dataframe(b.head(50), use_container_width=True, hide_index=True)
        st.download_button("Export all scores", b.to_csv(index=False).encode(), "dual_scores.csv", type="primary")

st.caption("ChurnGuard Enterprise · See DEMO.md")
