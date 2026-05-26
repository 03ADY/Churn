"""Plain-language interpretations for metrics and model outputs."""


def interpret_auc(auc: float) -> dict:
    if auc >= 0.80:
        level, tone = "Strong", "positive"
        meaning = "The model ranks likely churners much better than random guessing."
    elif auc >= 0.70:
        level, tone = "Good", "positive"
        meaning = "Useful for prioritizing retention campaigns; not perfect but actionable."
    elif auc >= 0.60:
        level, tone = "Fair", "warning"
        meaning = "Some signal exists; pair scores with business rules and human review."
    else:
        level, tone = "Weak", "warning"
        meaning = "Limited ranking power — improve features or collect more data before automating decisions."
    return {
        "level": level,
        "tone": tone,
        "meaning": meaning,
        "summary": f"**{level}** discriminator (AUC {auc:.2f}). {meaning}",
        "detail": (
            "**AUC (Area Under the ROC Curve)** measures how well the model orders customers from "
            "lowest to highest churn risk. 0.5 = random coin flip; 1.0 = perfect ranking. "
            "It does not depend on a single cutoff threshold, so it is reliable when churn is imbalanced."
        ),
    }


def interpret_f1(f1: float) -> dict:
    if f1 >= 0.60:
        level = "Strong"
        meaning = "When we flag churn at 50% risk, precision and recall are both solid."
    elif f1 >= 0.40:
        level = "Moderate"
        meaning = "Typical for churn at ~20% base rate — many teams still use AUC for targeting."
    else:
        level = "Low"
        meaning = "Hard trade-off at 50% cutoff; consider top-decile targeting instead of yes/no flags."
    return {
        "level": level,
        "meaning": meaning,
        "summary": f"**{level}** F1 ({f1:.2f}). {meaning}",
        "detail": (
            "**F1** balances precision (of flagged churners, how many actually leave) and "
            "recall (of all who leave, how many we caught). It is sensitive to the 0.5 threshold — "
            "a low F1 with a good AUC usually means you should contact the **top 10–20% risk** scores, not everyone above 50%."
        ),
    }


def interpret_agreement(rate: float) -> str:
    if rate >= 0.85:
        return (
            f"**{rate:.0%} agreement** — both models largely agree. You can present a single **ensemble score** "
            "to sales and support with confidence."
        )
    if rate >= 0.70:
        return (
            f"**{rate:.0%} agreement** — mostly aligned. Review cases where RF and NN disagree; "
            "those customers may need manual judgment."
        )
    return (
        f"**{rate:.0%} agreement** — models diverge on many customers. Treat scores as directional; "
        "investigate feature gaps or use the higher of the two probabilities for safety."
    )


def interpret_churn_rate(rate: float) -> str:
    return (
        f"**{rate:.1%} of customers in this file churned** (left the bank). "
        "That is your historical baseline — retention programs should aim to **reduce this rate** "
        "among high-value segments, not eliminate all predicted risk."
    )


def holdout_metric_cards(rf_auc: float, nn_auc: float, rf_f1: float, nn_f1: float) -> dict:
    """Single AUC/F1 narratives when RF and NN are close; split when they diverge."""
    auc_diff = abs(rf_auc - nn_auc)
    f1_diff = abs(rf_f1 - nn_f1)
    avg_auc = (rf_auc + nn_auc) / 2
    auc_interp = interpret_auc(avg_auc)
    rf_f1_x = interpret_f1(rf_f1)
    nn_f1_x = interpret_f1(nn_f1)

    if auc_diff < 0.02:
        auc_summary = (
            f"**Both models — {auc_interp['level'].lower()} ranking** "
            f"(RF AUC {rf_auc:.2f}, NN AUC {nn_auc:.2f}). {auc_interp['meaning']}"
        )
        auc_detail_once = auc_interp["detail"]
    else:
        auc_summary = None
        auc_detail_once = None

    if f1_diff < 0.05:
        f1_summary = (
            f"**At 50% cutoff** — RF F1 {rf_f1:.2f}, NN F1 {nn_f1:.2f}. {rf_f1_x['meaning']}"
        )
        f1_detail_once = rf_f1_x["detail"]
    else:
        f1_summary = None
        f1_detail_once = None

    return {
        "auc_combined": auc_diff < 0.02,
        "auc_summary": auc_summary,
        "auc_detail_once": auc_detail_once,
        "rf_auc_x": interpret_auc(rf_auc),
        "nn_auc_x": interpret_auc(nn_auc),
        "f1_combined": f1_diff < 0.05,
        "f1_summary": f1_summary,
        "f1_detail_once": f1_detail_once,
        "rf_f1_x": rf_f1_x,
        "nn_f1_x": nn_f1_x,
    }


def confusion_compare_note(rf_auc: float, nn_auc: float, rf_f1: float, nn_f1: float) -> str:
    if abs(rf_auc - nn_auc) >= 0.02 or abs(rf_f1 - nn_f1) < 0.03:
        return ""
    better = "Random Forest" if rf_f1 >= nn_f1 else "Neural Net"
    return (
        f"**Why do the confusion matrices differ?** AUC is almost the same, but **F1 uses a fixed 50% cutoff**. "
        f"**{better}** catches more churners at that cutoff (higher F1); the other is more conservative (fewer false alarms). "
        "For campaigns, rank by probability and contact the **top 10–20%**, not everyone above 50%."
    )


def compare_models(rf_auc: float, nn_auc: float) -> str:
    diff = abs(rf_auc - nn_auc)
    better = "Random Forest" if rf_auc >= nn_auc else "Neural Net"
    if diff < 0.02:
        return (
            "Both models perform **almost identically** on holdout data. "
            "**Random Forest** is easier to explain to stakeholders (feature rules); "
            "**Neural Net** can capture subtle non-linear patterns. For demos, cite both and use the **average score**."
        )
    return (
        f"**{better}** has slightly higher AUC ({max(rf_auc, nn_auc):.3f} vs {min(rf_auc, nn_auc):.3f}). "
        "Use it as the primary ranker, but keep the other model as a **second opinion** on borderline accounts."
    )


def confusion_narrative(cm, model_name: str) -> str:
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    total = tn + fp + fn + tp
    if total == 0:
        return ""
    return (
        f"**{model_name}** on holdout customers ({total:,} rows): "
        f"**{tp:,} true churners caught** (correctly flagged), "
        f"**{fn:,} missed** (left but predicted stay), "
        f"**{fp:,} false alarms** (predicted leave but stayed), "
        f"**{tn:,} correctly predicted stay**. "
        "False alarms waste outreach budget; misses are lost revenue — tune threshold to your cost of each error."
    )


def interpret_live_risk(rf_p: float, nn_p: float) -> dict:
    avg = (rf_p + nn_p) / 2
    if avg >= 0.70:
        band, action = "Very high", (
            "Prioritize **immediate outreach**: personal call, loyalty offer, or success manager assignment."
        )
    elif avg >= 0.50:
        band, action = "Elevated", (
            "Add to **retention campaign** within 7 days; monitor product usage and support tickets."
        )
    elif avg >= 0.30:
        band, action = "Moderate", (
            "Include in **nurture emails** and satisfaction surveys; no urgent intervention required."
        )
    else:
        band, action = "Lower", (
            "Standard service; focus budget on higher-risk segments."
        )
    disagree = abs(rf_p - nn_p) > 0.25
    extra = (
        " ⚠️ Models **disagree strongly** on this profile — have an agent review before acting."
        if disagree else ""
    )
    return {
        "band": band,
        "avg": avg,
        "summary": f"**{band} churn risk** ({avg:.0%} average probability). {action}{extra}",
        "detail": (
            f"**Random Forest** estimates **{rf_p:.0%}** chance this customer churns; "
            f"**Neural Net** estimates **{nn_p:.0%}**. "
            "These are probabilities from patterns in credit score, tenure, balance, geography, etc. — "
            "not certainties. Use them to **prioritize**, not to auto-close accounts."
        ),
        "playbook": action,
    }


def batch_export_help() -> str:
    return (
        "**RF_Prob / NN_Prob** — probability of churn (0–100%). "
        "**Ensemble_Prob** — average of both; good default for sorting a call list. "
        "**RF_Churn / NN_Churn** — yes/no at 50% threshold; use for CRM flags, but prefer top 10% by Ensemble_Prob for campaigns."
    )


def glossary_markdown() -> str:
    return """
### Models in this app
- **Random Forest (RF)** — many decision trees vote; strong on tabular bank data; easy to explain ("low tenure + high balance → risk").
- **Neural Net (NN)** — learns non-linear combinations of features; often matches RF on churn; good as a second opinion.

### Metrics
| Term | Meaning |
|------|---------|
| **AUC** | Ranking quality (who is riskier than whom). Best headline number. |
| **F1** | Balance of precision/recall at 50% cutoff. Often low when churn is rare. |
| **Holdout** | 20% of data the model never saw during training — fair test of performance. |
| **Churn rate** | % who already left in your CSV — historical fact, not a prediction. |

### What to do with scores
1. Sort by **Ensemble_Prob** descending.  
2. Contact **top 10–20%** with retention offers.  
3. Track whether contacted customers stay longer than a control group.
"""
