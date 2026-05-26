"""Dual-model churn: Random Forest + MLP (fast, Streamlit Cloud friendly)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from churn_dual.cloud import is_streamlit_cloud
from churn_dual.features import expected_columns_from_preprocessor, prepare_churn_df

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RF_PATH = MODEL_DIR / "rf_pipeline.joblib"
NN_PATH = MODEL_DIR / "nn_mlp.joblib"
PREP_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

# Cap training rows on Cloud for faster cold start
_CLOUD_TRAIN_CAP = 4000


def nn_predict_proba(nn: MLPClassifier, X: np.ndarray) -> np.ndarray:
    return nn.predict_proba(X)[:, 1]


def _training_subset(df: pd.DataFrame) -> pd.DataFrame:
    if is_streamlit_cloud() and len(df) > _CLOUD_TRAIN_CAP:
        return df.sample(_CLOUD_TRAIN_CAP, random_state=42)
    return df


def _try_load_cached(df: pd.DataFrame) -> dict | None:
    if not (RF_PATH.exists() and NN_PATH.exists() and PREP_PATH.exists()):
        return None
    try:
        rf = joblib.load(RF_PATH)
        nn = joblib.load(NN_PATH)
        prep = joblib.load(PREP_PATH)
    except Exception:
        # numpy / sklearn version mismatch between build and Cloud runtime
        return None
    expected = _load_feature_columns(prep)
    current = df.drop("Exited", axis=1).columns.tolist()
    if set(expected) != set(current):
        return None
    return _pack(df, rf, nn, prep, cached=True, feature_cols=expected)


def train_dual(df: pd.DataFrame, *, use_cache: bool = True) -> dict | None:
    df = prepare_churn_df(df)
    if "Exited" not in df.columns:
        return None

    # Bundled joblib often breaks across numpy versions (e.g. Py 3.13 on Cloud)
    if use_cache and not is_streamlit_cloud():
        cached = _try_load_cached(df)
        if cached is not None:
            return cached

    train_df = _training_subset(df)
    if len(train_df) < 50 or train_df["Exited"].nunique() < 2:
        return None

    X = train_df.drop("Exited", axis=1)
    y = train_df["Exited"]
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

    classes = np.array(sorted(y_train.unique()))
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = {int(classes[0]): float(cw[0]), int(classes[1]): float(cw[1])}

    n_trees = 50 if is_streamlit_cloud() else 80
    rf_best = Pipeline([
        ("preprocessor", pre),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=10,
                random_state=42,
                class_weight=cw_dict,
                n_jobs=1,
            ),
        ),
    ])
    rf_best.fit(X_train, y_train)

    fitted_pre = rf_best.named_steps["preprocessor"]
    Xtr = fitted_pre.transform(X_train)
    nn = MLPClassifier(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        max_iter=80 if is_streamlit_cloud() else 120,
        random_state=42,
    )
    nn.fit(Xtr, y_train)

    if not is_streamlit_cloud():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        feature_cols = X_train.columns.tolist()
        joblib.dump(rf_best, RF_PATH)
        joblib.dump(fitted_pre, PREP_PATH)
        joblib.dump(nn, NN_PATH)
        FEATURES_PATH.write_text(__import__("json").dumps(feature_cols), encoding="utf-8")

    return _pack(
        df, rf_best, nn, fitted_pre, cached=False,
        X_test=X_test, y_test=y_test, X_train=X_train, feature_cols=X_train.columns.tolist(),
    )


def _load_feature_columns(pre) -> list[str]:
    if FEATURES_PATH.exists():
        return __import__("json").loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return expected_columns_from_preprocessor(pre)


def _pack(df, rf, nn, pre, cached, X_test=None, y_test=None, X_train=None, feature_cols=None):
    if cached:
        X = df.drop("Exited", axis=1)
        y = df["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )
    cols = feature_cols or X_train.columns.tolist()
    X_test = X_test.reindex(columns=cols, fill_value=0)
    return {
        "rf": rf,
        "nn": nn,
        "pre": pre,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_cols": cols,
    }
