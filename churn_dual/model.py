"""Dual-model churn: Random Forest + MLP neural network (scikit-learn, cloud-friendly)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RF_PATH = MODEL_DIR / "rf_pipeline.joblib"
NN_PATH = MODEL_DIR / "nn_mlp.joblib"
PREP_PATH = MODEL_DIR / "preprocessor.joblib"


def nn_predict_proba(nn: MLPClassifier, X: np.ndarray) -> np.ndarray:
    return nn.predict_proba(X)[:, 1]


def train_dual(df: pd.DataFrame, *, use_cache: bool = True) -> dict | None:
    if use_cache and RF_PATH.exists() and NN_PATH.exists() and PREP_PATH.exists():
        rf = joblib.load(RF_PATH)
        nn = joblib.load(NN_PATH)
        prep = joblib.load(PREP_PATH)
        return _pack(df, rf, nn, prep, cached=True)

    if len(df) < 50 or df["Exited"].nunique() < 2:
        return None

    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classes = np.array(sorted(y_train.unique()))
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = {int(classes[0]): cw[0], int(classes[1]): cw[1]}

    rf_pipe = Pipeline([
        ("preprocessor", pre),
        ("classifier", RandomForestClassifier(random_state=42, class_weight=cw_dict)),
    ])
    rf_grid = GridSearchCV(
        rf_pipe,
        {"classifier__n_estimators": [100, 200], "classifier__max_depth": [5, 10]},
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=400,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    nn.fit(Xtr, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_best, RF_PATH)
    joblib.dump(pre, PREP_PATH)
    joblib.dump(nn, NN_PATH)
    return _pack(df, rf_best, nn, pre, cached=False, X_test=X_test, y_test=y_test, X_train=X_train)


def _pack(df, rf, nn, pre, cached, X_test=None, y_test=None, X_train=None):
    if cached:
        X = df.drop("Exited", axis=1)
        y = df["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {"rf": rf, "nn": nn, "pre": pre, "X_test": X_test, "y_test": y_test, "X_train_cols": X_train.columns.tolist()}
