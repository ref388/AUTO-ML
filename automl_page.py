"""
ML AutoPilot — AutoML Training Page (automl_page.py)
Upload a dataset, pick a target, benchmark many algorithms, tune the best,
inspect results, and persist everything to session_state for the other pages.

Loaded by app.py via runpy. DO NOT call st.set_page_config here — the root
app already does that.
"""

import io
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)

# Regressors
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Optional gradient boosting libraries
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Design tokens (light theme, matched to app shell)
# ══════════════════════════════════════════════════════════════════════════════

LIGHT_BG  = "#ffffff"
PANEL_BG  = "#ffffff"
BORDER    = "#e4e7ec"
INDIGO    = "#4f46e5"
TEAL      = "#14b8a6"
GREEN     = "#10b981"
AMBER     = "#f59e0b"
RED       = "#ef4444"
PURPLE    = "#8b5cf6"
TEXT      = "#1f2937"
SUBTEXT   = "#6b7280"


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* page-level overrides for this page only */
.aml-hero {
    background: linear-gradient(135deg, #ffffff 0%, #fafbff 100%);
    border: 1px solid #e4e7ec;
    border-left: 4px solid #4f46e5;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    position: relative;
    overflow: hidden;
}
.aml-hero::before {
    content: '';
    position: absolute; top: -30%; right: -5%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(79,70,229,0.10) 0%, transparent 60%);
}
.aml-hero h1 { color:#1f2937; font-size:1.9rem; font-weight:800; letter-spacing:-0.02em; margin:0 0 0.3rem; }
.aml-hero p  { color:#6b7280; margin:0; font-size:0.92rem; }
.aml-badge {
    display:inline-block; background:rgba(79,70,229,0.08);
    border:1px solid rgba(79,70,229,0.2); color:#4f46e5;
    font-family:'Space Mono',monospace; font-size:.68rem; font-weight:700;
    padding:.15rem .55rem; border-radius:999px; margin-right:.35rem; letter-spacing:.08em;
}

.aml-section {
    color:#6b7280; font-family:'Space Mono',monospace;
    font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.18em;
    border-bottom:1px solid #e4e7ec; padding-bottom:.5rem; margin:1.6rem 0 1rem;
}

/* metric cards */
.aml-metric-row { display:flex; gap:.8rem; flex-wrap:wrap; margin:1rem 0; }
.aml-metric {
    flex:1; min-width:120px;
    background:#ffffff; border:1px solid #e4e7ec; border-radius:10px;
    padding:1rem 1.2rem; text-align:center;
    box-shadow:0 1px 2px rgba(16,24,40,0.03);
}
.aml-metric .m-label {
    color:#6b7280; font-size:.68rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.12em;
    font-family:'Space Mono',monospace; margin-bottom:.35rem;
}
.aml-metric .m-value { color:#1f2937; font-size:1.4rem; font-weight:700; font-family:'Space Mono',monospace; }
.aml-metric .m-value.indigo { color:#4f46e5; }
.aml-metric .m-value.green  { color:#10b981; }
.aml-metric .m-value.amber  { color:#f59e0b; }
.aml-metric .m-value.red    { color:#ef4444; }
.aml-metric .m-value.purple { color:#8b5cf6; }

/* leaderboard */
.aml-lb-row {
    display:flex; align-items:center;
    background:#ffffff; border:1px solid #e4e7ec; border-radius:10px;
    padding:.7rem 1rem; margin-bottom:.4rem; gap:1rem;
    box-shadow:0 1px 2px rgba(16,24,40,0.03);
    transition: all .15s ease;
}
.aml-lb-row.best { border-color:#4f46e5; background:#fafbff; box-shadow:0 1px 2px rgba(79,70,229,0.12), 0 4px 12px rgba(79,70,229,0.08); }
.aml-lb-rank {
    font-family:'Space Mono',monospace; font-size:.9rem; font-weight:700;
    color:#9ca3af; width:30px; flex-shrink:0; text-align:center;
}
.aml-lb-rank.gold { color:#f59e0b; }
.aml-lb-name { color:#1f2937; font-weight:600; font-size:.95rem; flex:1; }
.aml-lb-name .lib-tag {
    display:inline-block; margin-left:.5rem; font-size:.65rem;
    font-family:'Space Mono',monospace; font-weight:700;
    padding:.1rem .4rem; border-radius:4px;
    background:rgba(20,184,166,0.1); color:#0d9488; letter-spacing:.06em;
}
.aml-lb-score {
    font-family:'Space Mono',monospace; font-size:.9rem;
    color:#4f46e5; font-weight:700;
}
.aml-lb-bar-wrap { width:130px; background:#f3f4f6; border-radius:4px; height:6px; }
.aml-lb-bar { height:6px; border-radius:4px; background:linear-gradient(90deg,#4f46e5,#8b5cf6); }
.aml-lb-bar.gold { background:linear-gradient(90deg,#f59e0b,#fbbf24); }

/* info / warn / good boxes */
.aml-info {
    background:rgba(79,70,229,0.06); border:1px solid rgba(79,70,229,0.18);
    border-radius:10px; padding:.85rem 1.2rem; color:#4b5563; font-size:.88rem; margin:.6rem 0;
}
.aml-info strong { color:#4f46e5; }
.aml-warn {
    background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.22);
    border-radius:10px; padding:.85rem 1.2rem; color:#4b5563; font-size:.88rem; margin:.4rem 0;
}
.aml-warn strong { color:#b45309; }
.aml-good {
    background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.22);
    border-radius:10px; padding:.85rem 1.2rem; color:#4b5563; font-size:.88rem; margin:.4rem 0;
}
.aml-good strong { color:#047857; }

/* task chip */
.aml-task-chip {
    display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(79,70,229,0.08); border:1px solid rgba(79,70,229,0.22);
    color:#4f46e5; font-family:'Space Mono',monospace; font-size:.78rem; font-weight:700;
    padding:.25rem .7rem; border-radius:999px;
}
.aml-task-chip.cls { background:rgba(20,184,166,0.08); border-color:rgba(20,184,166,0.22); color:#0d9488; }

/* drop zone */
.aml-dropzone {
    text-align:center; padding:3rem 2rem;
    background:#ffffff; border:2px dashed #d1d5db; border-radius:14px;
    margin-top:1rem; transition: all .2s ease;
}
.aml-dropzone:hover { border-color:#4f46e5; background:#fafbff; }
.aml-dropzone .icon { font-size:2.6rem; margin-bottom:.8rem; }
.aml-dropzone .text { font-size:1rem; color:#6b7280; }
.aml-dropzone .small { font-size:.78rem; margin-top:.5rem; font-family:'Space Mono',monospace; color:#9ca3af; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib light theme
# ══════════════════════════════════════════════════════════════════════════════

def mpl_light():
    plt.rcParams.update({
        "figure.facecolor":  LIGHT_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   SUBTEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "text.color":        TEXT,
        "grid.color":        "#f3f4f6",
        "grid.linewidth":    0.8,
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Task detection + preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def detect_task(y: pd.Series) -> str:
    """
    Auto-detect whether the target represents a regression or classification problem.

    Heuristics, in order:
      1. Object / category / bool dtype  → classification
      2. Numeric AND (≤15 unique values) AND (<5% unique ratio) AND
         (no single class dominates >95%)  → classification
      3. Otherwise → regression

    The dominance check catches datasets where a measurement is heavily
    concentrated on one value (e.g. 99% of rows have target=1.0). Treating
    those as classification gives a useless model that just predicts the
    majority class for everything.
    """
    if y.dtype == object or y.dtype.name == "category" or pd.api.types.is_bool_dtype(y):
        return "classification"
    n_unique = y.nunique(dropna=True)
    n_total  = max(len(y), 1)
    if n_unique <= 15 and (n_unique / n_total) < 0.05:
        # Check for extreme class dominance — if one value covers >95% of rows,
        # the variable is best modeled as continuous (or this is a broken target)
        top_share = y.value_counts(normalize=True).iloc[0] if n_unique > 0 else 1.0
        if top_share > 0.95:
            return "regression"
        return "classification"
    return "regression"


def split_columns(X: pd.DataFrame):
    """Return (numeric_cols, categorical_cols, datetime_cols)."""
    numeric, categorical, datetime_ = [], [], []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            # Try to parse as datetime — if >80% parse, treat as datetime
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    X[col] = parsed
                    datetime_.append(col)
                    continue
            except Exception:
                pass
            categorical.append(col)
    return numeric, categorical, datetime_


def expand_datetime_features(X: pd.DataFrame, dt_cols: list) -> pd.DataFrame:
    """Replace each datetime column with year/month/day/dayofweek/quarter."""
    for col in dt_cols:
        dt = pd.to_datetime(X[col], errors="coerce")
        X[f"{col}_year"]      = dt.dt.year.fillna(0).astype(int)
        X[f"{col}_month"]     = dt.dt.month.fillna(0).astype(int)
        X[f"{col}_day"]       = dt.dt.day.fillna(0).astype(int)
        X[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        X[f"{col}_quarter"]   = dt.dt.quarter.fillna(0).astype(int)
        X = X.drop(columns=[col])
    return X


def build_preprocessor(numeric_cols: list, cat_cols: list,
                       cat_strategy: str = "auto", high_card_threshold: int = 15):
    """
    Construct a ColumnTransformer that handles numeric + categorical preprocessing.

    cat_strategy:
      'auto'   - OneHot for low-cardinality (≤ threshold), Ordinal for the rest
      'onehot' - always OneHot (dangerous for high-cardinality)
      'ordinal'- always Ordinal (label encoding)
    """
    transformers = []

    if numeric_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if cat_cols:
        # Determined per-column at construction time, but ColumnTransformer needs
        # static column lists — so we split cat_cols by cardinality here.
        if cat_strategy == "auto":
            # caller passes already-classified columns; ignore threshold
            pass

        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False, max_categories=20)
                       if cat_strategy in ("onehot", "auto")
                       else OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def preprocess(df: pd.DataFrame, target: str,
               cat_strategy: str = "auto",
               drop_threshold: float = 0.6):
    """
    Full preprocessing flow. Returns:
        X_processed      — numpy array ready for any model
        y                — target (label-encoded if classification)
        task             — 'regression' | 'classification'
        feature_names    — list of feature names AFTER expansion
        preprocessor     — fitted ColumnTransformer (for later use)
        class_labels     — list of original class names (classification only)
    """
    # Coerce ALL column names to strings — sklearn's ColumnTransformer
    # interprets non-string column lists as positional indices, which
    # crashes when column names are integers (e.g. Excel files with
    # numeric headers like 1, 2, 3...).
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    target = str(target)

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop columns with too many missing values
    miss_frac = X.isnull().mean()
    drop_cols = miss_frac[miss_frac > drop_threshold].index.tolist()
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Detect column types
    num_cols, cat_cols, dt_cols = split_columns(X)

    # Expand datetime → numeric pieces
    if dt_cols:
        X = expand_datetime_features(X, dt_cols)
        # Re-classify newly created cols
        num_cols, cat_cols, _ = split_columns(X)

    # Drop high-cardinality "ID-like" categorical columns (>50% unique)
    id_like = []
    for col in list(cat_cols):
        if X[col].nunique() / max(len(X), 1) > 0.5 and X[col].nunique() > 20:
            id_like.append(col)
    if id_like:
        X = X.drop(columns=id_like)
        cat_cols = [c for c in cat_cols if c not in id_like]

    pre = build_preprocessor(num_cols, cat_cols, cat_strategy=cat_strategy)
    X_proc = pre.fit_transform(X)

    # Build feature names after one-hot expansion
    try:
        feature_names = list(pre.get_feature_names_out())
        # Strip the transformer prefix (e.g. "num__age" → "age")
        feature_names = [f.split("__", 1)[-1] for f in feature_names]
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_proc.shape[1])]

    # Encode target
    task = detect_task(y)
    class_labels = None
    if task == "classification":
        y_str = y.astype(str)
        class_labels = sorted(y_str.unique().tolist())
        label_map = {lbl: i for i, lbl in enumerate(class_labels)}
        y_arr = y_str.map(label_map).values
    else:
        y_arr = pd.to_numeric(y, errors="coerce").values
        # Drop rows where target couldn't be parsed
        mask = ~np.isnan(y_arr)
        X_proc = X_proc[mask]
        y_arr = y_arr[mask]

    return {
        "X":             pd.DataFrame(X_proc, columns=feature_names),
        "y":             y_arr,
        "task":          task,
        "feature_names": feature_names,
        "preprocessor":  pre,
        "class_labels":  class_labels,
        "raw_X_columns": X.columns.tolist(),
        "dropped_cols":  drop_cols + id_like,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm catalog — includes optional XGBoost / LightGBM / CatBoost
# ══════════════════════════════════════════════════════════════════════════════

def regressors(class_weight=None, n_jobs=1, fast: bool = False):
    """
    Return dict of {name: (estimator, library_tag)}.

    n_jobs=1 by default: parallelism is handled by the outer cross_val_score.
    Nested n_jobs=-1 deadlocks on small instances (Streamlit Cloud, 1 CPU).
    Pass n_jobs=-1 only when calling outside cross_val_score.

    fast=True skips slow algorithms (SVR — quadratic in n_samples).
    """
    cat = {
        "Ridge":                  (Ridge(),                                              "sklearn"),
        "Lasso":                  (Lasso(max_iter=5000),                                 "sklearn"),
        "ElasticNet":             (ElasticNet(max_iter=5000),                            "sklearn"),
        "Decision Tree":          (DecisionTreeRegressor(random_state=42),               "sklearn"),
        "K-Nearest Neighbors":    (KNeighborsRegressor(),                                "sklearn"),
        "Random Forest":          (RandomForestRegressor(n_estimators=200, n_jobs=n_jobs,
                                                          random_state=42),              "sklearn"),
        "Extra Trees":            (ExtraTreesRegressor(n_estimators=200, n_jobs=n_jobs,
                                                        random_state=42),                "sklearn"),
        "Gradient Boosting":      (GradientBoostingRegressor(n_estimators=200,
                                                              random_state=42),          "sklearn"),
        "Hist Gradient Boosting": (HistGradientBoostingRegressor(random_state=42),       "sklearn"),
    }
    if not fast:
        cat["SVR (RBF)"] = (SVR(kernel="rbf"), "sklearn")

    if XGB_AVAILABLE:
        cat["XGBoost"]  = (XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                                        n_jobs=n_jobs, random_state=42,
                                        verbosity=0, tree_method="hist"),    "xgboost")
    if LGBM_AVAILABLE:
        cat["LightGBM"] = (LGBMRegressor(n_estimators=300, learning_rate=0.1,
                                          n_jobs=n_jobs, random_state=42,
                                          verbosity=-1),                     "lightgbm")
    if CATBOOST_AVAILABLE:
        cat["CatBoost"] = (CatBoostRegressor(iterations=300, learning_rate=0.1,
                                              random_state=42, verbose=False,
                                              thread_count=n_jobs),           "catboost")
    return cat


def classifiers(class_weight=None, n_jobs=1, fast: bool = False):
    """See regressors() docstring for the n_jobs / fast behaviour."""
    cat = {
        "Logistic Regression":    (LogisticRegression(max_iter=2000, n_jobs=n_jobs,
                                                       random_state=42,
                                                       class_weight=class_weight),       "sklearn"),
        "Decision Tree":          (DecisionTreeClassifier(random_state=42,
                                                           class_weight=class_weight),    "sklearn"),
        "K-Nearest Neighbors":    (KNeighborsClassifier(n_jobs=n_jobs),                  "sklearn"),
        "Random Forest":          (RandomForestClassifier(n_estimators=200, n_jobs=n_jobs,
                                                           random_state=42,
                                                           class_weight=class_weight),    "sklearn"),
        "Extra Trees":            (ExtraTreesClassifier(n_estimators=200, n_jobs=n_jobs,
                                                         random_state=42,
                                                         class_weight=class_weight),      "sklearn"),
        "Gradient Boosting":      (GradientBoostingClassifier(n_estimators=200,
                                                               random_state=42),          "sklearn"),
        "Hist Gradient Boosting": (HistGradientBoostingClassifier(random_state=42,
                                                                   class_weight=class_weight),
                                   "sklearn"),
    }
    if not fast:
        cat["SVC (RBF)"] = (SVC(kernel="rbf", probability=True, random_state=42,
                                 class_weight=class_weight), "sklearn")

    if XGB_AVAILABLE:
        cat["XGBoost"]  = (XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                                          n_jobs=n_jobs, random_state=42, verbosity=0,
                                          tree_method="hist",
                                          eval_metric="logloss"),
                           "xgboost")
    if LGBM_AVAILABLE:
        cat["LightGBM"] = (LGBMClassifier(n_estimators=300, learning_rate=0.1,
                                           n_jobs=n_jobs, random_state=42,
                                           class_weight=class_weight, verbosity=-1),
                           "lightgbm")
    if CATBOOST_AVAILABLE:
        cat["CatBoost"] = (CatBoostClassifier(iterations=300, learning_rate=0.1,
                                                random_state=42, verbose=False,
                                                thread_count=n_jobs),
                           "catboost")
    return cat


# Param grids for tuning the most useful algorithms
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators":      [100, 200, 400, 800],
        "max_depth":         [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.5, 0.8],
    },
    "Extra Trees": {
        "n_estimators":      [100, 200, 400],
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2", 0.5],
    },
    "Gradient Boosting": {
        "n_estimators":  [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth":     [2, 3, 4, 6],
        "subsample":     [0.6, 0.8, 1.0],
        "min_samples_split": [2, 5, 10],
    },
    "Hist Gradient Boosting": {
        "max_iter":      [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth":     [None, 4, 6, 10],
        "min_samples_leaf": [10, 20, 50],
        "l2_regularization": [0, 0.1, 1.0],
    },
    "XGBoost": {
        "n_estimators":  [100, 300, 500],
        "max_depth":     [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "LightGBM": {
        "n_estimators":   [100, 300, 500],
        "num_leaves":     [15, 31, 63, 127],
        "learning_rate":  [0.01, 0.05, 0.1, 0.2],
        "subsample":      [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [5, 20, 50],
    },
    "CatBoost": {
        "iterations":    [100, 300, 500],
        "depth":         [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg":   [1, 3, 5, 7],
    },
    "Ridge":      {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    "Lasso":      {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 10, 15], "weights": ["uniform", "distance"], "p": [1, 2]},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"], "solver": ["lbfgs"]},
    "Decision Tree": {"max_depth": [None, 5, 10, 20, 30], "min_samples_split": [2, 5, 10, 20]},
    "SVR (RBF)":   {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"], "epsilon": [0.01, 0.1, 0.5]},
    "SVC (RBF)":   {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
}


# ══════════════════════════════════════════════════════════════════════════════
# Benchmarking
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(X_train, y_train, task, cv=5, fast=False, class_weight=None,
              progress_cb=None, parallel_cv: bool = True):
    """
    Cross-validate each candidate model and return their scores.

    parallel_cv:
      True  → cross_val_score uses n_jobs=-1 to parallelize folds (faster, more RAM)
      False → cross_val_score uses n_jobs=1 (safer on tiny instances)

    Each model is wrapped in a try/except: failures are recorded but never crash
    the whole run. This is important on free hosting tiers where memory limits
    can kill specific models (gradient-boosting libraries are the usual offenders).
    """
    catalog = (regressors(fast=fast) if task == "regression"
               else classifiers(fast=fast, class_weight=class_weight))
    scoring = "r2" if task == "regression" else "f1_weighted"
    cv_jobs = -1 if parallel_cv else 1

    results = {}
    n = len(catalog)
    for i, (name, (mdl, lib)) in enumerate(catalog.items()):
        if progress_cb:
            progress_cb(i / n, name)
        try:
            scores = cross_val_score(mdl, X_train, y_train,
                                      cv=cv, scoring=scoring, n_jobs=cv_jobs,
                                      error_score="raise")
            results[name] = {
                "mean":  float(scores.mean()),
                "std":   float(scores.std()),
                "scores": scores.tolist(),
                "lib":   lib,
                "ok":    True,
            }
        except MemoryError:
            results[name] = {"mean": -np.inf, "std": 0, "scores": [],
                              "lib": lib, "ok": False,
                              "error": "Out of memory — try Fast mode or upgrade your hosting."}
        except Exception as e:
            # Catch-all: keep going even if one library has a binary incompatibility,
            # native crash, etc. This is what prevents the entire run from dying
            # when (e.g.) XGBoost segfaults on a particular dataset.
            err_msg = str(e)[:120] if str(e) else f"{type(e).__name__}"
            results[name] = {"mean": -np.inf, "std": 0, "scores": [],
                              "lib": lib, "ok": False, "error": err_msg}
        if progress_cb:
            progress_cb((i + 1) / n, name)
    return results


def tune(X_train, y_train, best_name, task, n_iter=40, cv=5,
         class_weight=None, parallel: bool = True):
    """
    Hyperparameter tuning via RandomizedSearchCV.
    parallel=False uses n_jobs=1, safer for big datasets on small instances.
    """
    catalog = (regressors() if task == "regression"
               else classifiers(class_weight=class_weight))
    mdl, lib = catalog[best_name]
    grid = PARAM_GRIDS.get(best_name, {})
    scoring = "r2" if task == "regression" else "f1_weighted"

    if not grid:
        mdl.fit(X_train, y_train)
        return mdl, {}

    rs = RandomizedSearchCV(
        mdl, grid, n_iter=n_iter, cv=cv,
        scoring=scoring, random_state=42,
        n_jobs=-1 if parallel else 1,
        error_score=np.nan,
    )
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred):
    return {
        "MAE":    mean_absolute_error(y_true, y_pred),
        "RMSE":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²":     r2_score(y_true, y_pred),
        "MAPE %": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100),
    }


def classification_metrics(y_true, y_pred, y_prob=None):
    out = {
        "Accuracy":      accuracy_score(y_true, y_pred),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                out["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                out["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                                average="weighted")
        except Exception:
            pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def parity_plot(y_train, yp_train, y_test, yp_test, target_name):
    mpl_light()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(LIGHT_BG)
    for ax, (yt, yp, label, colour) in zip(axes, [
        (y_train, yp_train, "Train", INDIGO),
        (y_test,  yp_test,  "Test",  GREEN),
    ]):
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "--", color=AMBER, lw=1.5, alpha=0.8, label="Ideal")
        ax.scatter(yt, yp, alpha=0.55, s=22, color=colour, edgecolors="none", label=label)
        r2 = r2_score(yt, yp)
        ax.set_title(f"{label}  ·  R² = {r2:.4f}", fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlabel(f"Actual {target_name}", fontsize=9)
        ax.set_ylabel(f"Predicted {target_name}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
    fig.suptitle("Parity Plot — Predicted vs Actual", fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    return fig


def residual_plot(y_train, yp_train, y_test, yp_test):
    mpl_light()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(LIGHT_BG)
    for ax, (yt, yp, label, colour) in zip(axes, [
        (y_train, yp_train, "Train", INDIGO),
        (y_test,  yp_test,  "Test",  GREEN),
    ]):
        res = yt - yp
        ax.axhline(0, color=AMBER, lw=1.5, linestyle="--", alpha=0.8)
        ax.scatter(yp, res, alpha=0.5, s=20, color=colour, edgecolors="none")
        ax.set_title(f"{label} Residuals", fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.grid(True, alpha=0.4)
    fig.suptitle("Residual Plot", fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    return fig


def confusion_matrix_plot(y_true, y_pred, labels):
    mpl_light()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7), max(4, len(labels) * 0.6)))
    fig.patch.set_facecolor(LIGHT_BG)

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    # Annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else TEXT,
                    fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix (Test set)", fontsize=11, fontweight="bold", color=TEXT)

    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig


def feature_importance_plot(importances: pd.Series, top_n: int = 15):
    mpl_light()
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(3, len(top) * 0.4)))
    fig.patch.set_facecolor(LIGHT_BG)

    colours = [AMBER if i >= len(top) - 3 else INDIGO for i in range(len(top))]
    ax.barh(top.index, top.values, color=colours, alpha=0.88, edgecolor="none")
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title(f"Top {len(top)} feature importances (model-native)",
                 fontsize=11, fontweight="bold", color=TEXT)
    ax.grid(True, alpha=0.4, axis="x")

    for i, v in enumerate(top.values):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8, color=SUBTEXT)

    fig.tight_layout()
    return fig


def class_distribution_plot(y, labels):
    mpl_light()
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(LIGHT_BG)

    bars = ax.bar([labels[i] for i in counts.index], counts.values,
                  color=INDIGO, alpha=0.85, edgecolor="none")
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f" {n:,}",
                ha="center", va="bottom", fontsize=9, color=SUBTEXT)
    ax.set_title("Class distribution", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_ylabel("Count", fontsize=9)
    ax.grid(True, alpha=0.4, axis="y")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HTML helpers
# ══════════════════════════════════════════════════════════════════════════════

def metric_cards_html(metrics: dict) -> str:
    colour_map = {
        "R²": "indigo", "Accuracy": "indigo", "F1 (weighted)": "green",
        "ROC-AUC": "green", "MAE": "amber", "RMSE": "amber", "MAPE %": "red",
    }
    cards = ""
    for k, v in metrics.items():
        if v is None:
            continue
        colour = colour_map.get(k, "")
        if isinstance(v, float):
            fmt = f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
        else:
            fmt = str(v)
        cards += f"""
        <div class="aml-metric">
            <div class="m-label">{k}</div>
            <div class="m-value {colour}">{fmt}</div>
        </div>"""
    return f"<div class='aml-metric-row'>{cards}</div>"


def leaderboard_html(results: dict, task: str, best_name: str) -> str:
    metric_label = "CV R²" if task == "regression" else "CV F1"
    items = [(k, v) for k, v in results.items() if v["ok"]]
    items.sort(key=lambda x: x[1]["mean"], reverse=True)

    if not items:
        return "<div class='aml-warn'>No model finished successfully.</div>"

    best_score = items[0][1]["mean"]
    html = f"<div class='aml-section'>{metric_label} Leaderboard</div>"
    for rank, (name, info) in enumerate(items, 1):
        pct = max(0, min(100, info["mean"] / (best_score + 1e-9) * 100))
        rank_cls = "gold" if rank == 1 else ""
        bar_cls  = "gold" if rank == 1 else ""
        row_cls  = "best" if name == best_name else ""
        rank_sym = "🥇" if rank == 1 else f"{rank:02d}"
        lib_tag  = f"<span class='lib-tag'>{info['lib']}</span>" if info["lib"] != "sklearn" else ""
        html += f"""
        <div class='aml-lb-row {row_cls}'>
            <div class='aml-lb-rank {rank_cls}'>{rank_sym}</div>
            <div class='aml-lb-name'>{name}{lib_tag}</div>
            <div style='flex:1'><div class='aml-lb-bar-wrap'><div class='aml-lb-bar {bar_cls}' style='width:{pct:.0f}%'></div></div></div>
            <div class='aml-lb-score'>{info['mean']:.4f} ± {info['std']:.4f}</div>
        </div>"""

    # Failed models — surface the real reason so users can act on it
    failed = [(k, v) for k, v in results.items() if not v["ok"]]
    if failed:
        rows = ""
        for n, v in failed:
            err = v.get("error", "unknown error")
            rows += f"<div style='margin-top:.3rem;font-size:.82rem;'>• <strong>{n}</strong>: <span style='color:#9ca3af;font-family:Space Mono,monospace;'>{err}</span></div>"
        html += f"<div class='aml-warn' style='margin-top:.5rem;'>⚠️ <strong>{len(failed)}</strong> algorithm(s) failed and were skipped:{rows}</div>"

    return html


# ══════════════════════════════════════════════════════════════════════════════
# Page UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────────
extras = []
if XGB_AVAILABLE:      extras.append("XGBoost")
if LGBM_AVAILABLE:     extras.append("LightGBM")
if CATBOOST_AVAILABLE: extras.append("CatBoost")
extra_str = " · ".join(extras) if extras else "scikit-learn only"

st.markdown(f"""
<div class="aml-hero">
  <h1>🤖 Train your model</h1>
  <p>
    <span class="aml-badge">UPLOAD</span>
    <span class="aml-badge">BENCHMARK</span>
    <span class="aml-badge">TUNE</span>
    <span class="aml-badge">EVALUATE</span>
    &nbsp; Upload your dataset, pick a target, get the best model automatically.<br>
    <span style='color:#9ca3af; font-size:.8rem; font-family:Space Mono,monospace;'>
      Extras detected: {extra_str}
    </span>
  </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Training settings")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05,
                          help="Fraction of rows held out for testing.")
    cv_folds  = st.slider("Cross-validation folds", 3, 10, 5)

    fast_mode = st.checkbox("⚡ Fast mode (skip slow algorithms)",
                            value=False,
                            help="Skip SVR/SVC. Strongly recommended for >5K rows.")

    st.markdown("---")
    st.markdown("### 🎯 Task override")
    task_override = st.selectbox(
        "Task type (or auto-detect)",
        ["Auto-detect", "Regression", "Classification"],
        index=0,
        help="If auto-detection gets it wrong, force the right one here."
    )

    st.markdown("---")
    st.markdown("### 🔧 Hyperparameter tuning")
    do_tune = st.checkbox("Enable RandomizedSearchCV", value=False,
                          help="Tunes the best model — slower but usually +1-5% accuracy.")
    n_iter  = st.slider("Tuning iterations", 10, 100, 40, 10, disabled=not do_tune)

    st.markdown("---")
    st.markdown("### 📊 Preprocessing")
    cat_strategy = st.selectbox(
        "Categorical encoding",
        ["auto (one-hot, max 20 cats)", "ordinal (label codes)"],
        index=0,
    )
    cat_strategy = "auto" if cat_strategy.startswith("auto") else "ordinal"

    st.markdown("---")
    st.markdown("### 📈 Display")
    show_residuals = st.checkbox("Show residual plot (regression)", value=True)
    show_importance = st.checkbox("Show feature importance", value=True)


# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your dataset (.xlsx, .xls, or .csv)",
    type=["xlsx", "xls", "csv"],
    help="First row should be column headers.",
    key="aml_uploader",
)

# Re-use df from session if available (e.g. from Data Explorer)
session_df = st.session_state.get("df") if uploaded is None else None

if uploaded is None and not isinstance(session_df, pd.DataFrame):
    st.markdown("""
    <div class="aml-dropzone">
        <div class="icon">📂</div>
        <div class="text">Drop an Excel or CSV file to begin</div>
        <div class="small">.xlsx · .xls · .csv</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.lower().endswith(".xls"):
            df = pd.read_excel(uploaded, engine="xlrd")
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")
        filename = uploaded.name
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
else:
    df = session_df.copy()
    filename = st.session_state.get("filename", "session_dataset")

# Quick dataset summary card
n_missing = int(df.isnull().sum().sum())
miss_pct  = round(n_missing / df.size * 100, 1) if df.size else 0

st.markdown(f"""
<div class="aml-info">
    <strong>{filename}</strong> &nbsp;·&nbsp;
    {df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns &nbsp;·&nbsp;
    {n_missing:,} missing values ({miss_pct}%)
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 Preview data (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)


# ── Target column ────────────────────────────────────────────────────────────
st.markdown('<div class="aml-section">Target column</div>', unsafe_allow_html=True)
target = st.selectbox(
    "What do you want to predict?",
    options=df.columns.tolist(),
    index=len(df.columns) - 1,
    help="The output variable your model will learn to predict.",
)

# Validate target
if df[target].isnull().mean() > 0.3:
    st.error(f"Target column '{target}' has >30% missing values. Pick another column or clean your data first.")
    st.stop()

df = df.dropna(subset=[target]).reset_index(drop=True)


# ── Preprocess ───────────────────────────────────────────────────────────────
try:
    prep = preprocess(df, target, cat_strategy=cat_strategy)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

X            = prep["X"]
y            = prep["y"]
auto_task    = prep["task"]
feature_names = prep["feature_names"]
class_labels = prep["class_labels"]

# Override if user asked
if task_override == "Regression":
    task = "regression"
elif task_override == "Classification":
    task = "classification"
    if class_labels is None:   # was auto-detected as regression — convert
        unique = sorted(pd.Series(y).unique().tolist())
        class_labels = [str(u) for u in unique]
        y = pd.Series(y).map({v: i for i, v in enumerate(unique)}).values
else:
    task = auto_task

# Display detected task
task_chip_cls = "cls" if task == "classification" else ""
st.markdown(f"""
<div class="aml-info">
    Task: <span class="aml-task-chip {task_chip_cls}">{task.upper()}</span>
    &nbsp;·&nbsp; <strong>{X.shape[1]}</strong> features after encoding
    &nbsp;·&nbsp; <strong>{len(y):,}</strong> samples
    {f" &nbsp;·&nbsp; <strong>{len(class_labels)}</strong> classes" if class_labels else ""}
    {f" &nbsp;·&nbsp; {len(prep['dropped_cols'])} column(s) auto-dropped" if prep["dropped_cols"] else ""}
</div>
""", unsafe_allow_html=True)

if prep["dropped_cols"]:
    with st.expander(f"ℹ️ {len(prep['dropped_cols'])} columns were auto-dropped"):
        st.write(", ".join(prep["dropped_cols"]))
        st.caption("Dropped because they were >60% missing or looked like ID/text columns.")


# ── Class imbalance check ────────────────────────────────────────────────────
class_weight = None
if task == "classification":
    cls_counts = pd.Series(y).value_counts()
    if len(cls_counts) >= 2:
        imbalance = cls_counts.max() / cls_counts.min()
        if imbalance > 5:
            st.markdown(f"""
            <div class="aml-warn">
                ⚠️ <strong>Class imbalance detected</strong> ({imbalance:.1f}× ratio
                between largest and smallest class). Auto-enabling <code>class_weight='balanced'</code>
                so the model doesn't ignore minority classes.
            </div>
            """, unsafe_allow_html=True)
            class_weight = "balanced"

    with st.expander("📊 Class distribution"):
        fig = class_distribution_plot(y, class_labels)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── Train / test split ───────────────────────────────────────────────────────
stratify_y = None
if task == "classification":
    try:
        if pd.Series(y).value_counts().min() >= 2:
            stratify_y = y
    except Exception:
        pass

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=stratify_y,
)


# ── Run button ───────────────────────────────────────────────────────────────
n_rows = len(X)
if n_rows > 5000 and not fast_mode:
    st.markdown(f"""
    <div class="aml-warn">
        💡 You have <strong>{n_rows:,}</strong> rows. Consider enabling
        <strong>⚡ Fast mode</strong> in the sidebar — it skips SVM (very slow on big data)
        and finishes in a fraction of the time.
    </div>
    """, unsafe_allow_html=True)

run_btn = st.button("🚀 Run AutoPilot", use_container_width=True, type="primary")

if not run_btn:
    st.markdown("""
    <div style='text-align:center;padding:2rem;color:#9ca3af;
                font-family:Space Mono,monospace;font-size:.82rem;'>
        Configure settings in the sidebar, then click <strong>Run AutoPilot</strong> ↑
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Benchmark ────────────────────────────────────────────────────────────────
st.markdown('<div class="aml-section">Benchmarking algorithms</div>', unsafe_allow_html=True)

# Heuristic: disable parallel CV when the dataset is large enough that running
# 5 folds × multiple cores × gradient-boosting libraries blows past 1GB RAM
# (which is the Streamlit Cloud free-tier limit).
n_rows_train = len(X_train)
parallel_cv  = n_rows_train < 10_000
if not parallel_cv:
    st.markdown(f"""
    <div class="aml-info">
        🧠 Dataset has {n_rows_train:,} training rows — using sequential cross-validation
        to stay within memory limits. This is slower but won't crash on free hosting tiers.
    </div>
    """, unsafe_allow_html=True)

progress = st.progress(0.0, text="Starting benchmark…")

def _progress_cb(p, name):
    progress.progress(p, text=f"Evaluating {name}…  ({int(p*100)}%)")

results = benchmark(X_train, y_train, task,
                    cv=cv_folds, fast=fast_mode,
                    class_weight=class_weight,
                    progress_cb=_progress_cb,
                    parallel_cv=parallel_cv)
progress.empty()

# Pick best model
ok_results = {k: v for k, v in results.items() if v["ok"]}
if not ok_results:
    failed_msgs = "\n".join(f"  • {n}: {v.get('error', '?')}"
                              for n, v in results.items() if not v["ok"])
    st.error(f"All algorithms failed. First few errors:\n{failed_msgs}")
    st.stop()

best_name = max(ok_results, key=lambda k: ok_results[k]["mean"])
st.markdown(leaderboard_html(results, task, best_name), unsafe_allow_html=True)

st.markdown(f"""
<div class="aml-info">
    Best algorithm: <strong>{best_name}</strong> &nbsp;·&nbsp;
    CV score = <strong>{results[best_name]['mean']:.4f}</strong>
    {' &nbsp;·&nbsp; Running RandomizedSearchCV tuning…' if do_tune else ''}
</div>
""", unsafe_allow_html=True)


# ── Tune ─────────────────────────────────────────────────────────────────────
best_params = {}
if do_tune:
    with st.spinner(f"🔧 Tuning {best_name} with {n_iter} iterations…"):
        try:
            best_model, best_params = tune(X_train, y_train, best_name, task,
                                            n_iter=n_iter, cv=cv_folds,
                                            class_weight=class_weight,
                                            parallel=parallel_cv)
        except Exception as e:
            st.error(f"Tuning failed, falling back to default params: {e}")
            cat = regressors() if task == "regression" else classifiers(class_weight=class_weight)
            best_model = cat[best_name][0]
            best_model.fit(X_train, y_train)
    if best_params:
        param_str = " &nbsp;·&nbsp; ".join(f"<strong>{k}</strong>={v}" for k, v in best_params.items())
        st.markdown(f"<div class='aml-good'>✅ Best params found: {param_str}</div>", unsafe_allow_html=True)
else:
    cat = regressors() if task == "regression" else classifiers(class_weight=class_weight)
    best_model = cat[best_name][0]
    best_model.fit(X_train, y_train)


# ── Persist to session_state for other pages ─────────────────────────────────
st.session_state.model         = best_model
st.session_state.X_train       = X_train
st.session_state.X_test        = X_test
st.session_state.y_train       = y_train
st.session_state.y_test        = y_test
st.session_state.task          = task
st.session_state.target        = target
st.session_state.feature_names = feature_names
st.session_state.class_labels  = class_labels
st.session_state.preprocessor  = prep["preprocessor"]
st.session_state.df            = df.copy()
st.session_state.filename      = filename
st.session_state.best_name     = best_name


# ── Evaluate ─────────────────────────────────────────────────────────────────
yp_train = best_model.predict(X_train)
yp_test  = best_model.predict(X_test)

st.markdown('<div class="aml-section">Performance metrics</div>', unsafe_allow_html=True)
col_tr, col_te = st.columns(2)

if task == "regression":
    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr  = np.asarray(y_test,  dtype=float)

    with col_tr:
        st.markdown("**Train**")
        st.markdown(metric_cards_html(regression_metrics(y_train_arr, yp_train)),
                    unsafe_allow_html=True)
    with col_te:
        st.markdown("**Test**")
        st.markdown(metric_cards_html(regression_metrics(y_test_arr, yp_test)),
                    unsafe_allow_html=True)

    st.markdown('<div class="aml-section">Parity plot</div>', unsafe_allow_html=True)
    fig = parity_plot(y_train_arr, yp_train, y_test_arr, yp_test, target)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if show_residuals:
        st.markdown('<div class="aml-section">Residual plot</div>', unsafe_allow_html=True)
        fig = residual_plot(y_train_arr, yp_train, y_test_arr, yp_test)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

else:
    y_prob_train = best_model.predict_proba(X_train) if hasattr(best_model, "predict_proba") else None
    y_prob_test  = best_model.predict_proba(X_test)  if hasattr(best_model, "predict_proba") else None

    with col_tr:
        st.markdown("**Train**")
        st.markdown(metric_cards_html(classification_metrics(y_train, yp_train, y_prob_train)),
                    unsafe_allow_html=True)
    with col_te:
        st.markdown("**Test**")
        st.markdown(metric_cards_html(classification_metrics(y_test, yp_test, y_prob_test)),
                    unsafe_allow_html=True)

    st.markdown('<div class="aml-section">Confusion matrix</div>', unsafe_allow_html=True)
    fig = confusion_matrix_plot(y_test, yp_test, class_labels)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("📋 Full classification report (test set)"):
        report = classification_report(y_test, yp_test,
                                        target_names=class_labels,
                                        output_dict=False)
        st.code(report, language="text")


# ── Feature importance ───────────────────────────────────────────────────────
if show_importance:
    importances = None
    try:
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index=feature_names)
        elif hasattr(best_model, "coef_"):
            coef = best_model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            importances = pd.Series(np.abs(coef), index=feature_names)
    except Exception:
        importances = None

    if importances is not None and not importances.isnull().all():
        importances = importances.sort_values(ascending=False)
        st.markdown('<div class="aml-section">Feature importance (model-native)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="aml-info">
            Model-native importance — fast but less precise than SHAP.
            For a deeper view per row, visit the <strong>Model Explainability</strong> page.
        </div>
        """, unsafe_allow_html=True)
        fig = feature_importance_plot(importances, top_n=15)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── Downloads ────────────────────────────────────────────────────────────────
st.markdown("---")

# Predictions Excel
pred_df = pd.DataFrame({
    "Actual":    np.concatenate([np.asarray(y_train), np.asarray(y_test)]),
    "Predicted": np.concatenate([yp_train, yp_test]),
    "Split":     ["Train"] * len(y_train) + ["Test"] * len(y_test),
})
if class_labels is not None:
    pred_df["Actual"]    = pd.Categorical.from_codes(pred_df["Actual"].astype(int), class_labels)
    pred_df["Predicted"] = pd.Categorical.from_codes(pred_df["Predicted"].astype(int), class_labels)

buf = io.BytesIO()
pred_df.to_excel(buf, index=False, engine="openpyxl")

# Trained model pickle bundle (model + preprocessor + metadata)
model_bundle = {
    "model":         best_model,
    "preprocessor":  prep["preprocessor"],
    "feature_names": feature_names,
    "task":          task,
    "target":        target,
    "class_labels":  class_labels,
    "best_name":     best_name,
    "best_params":   best_params,
    "trained_at":    datetime.utcnow().isoformat(),
}
pkl_buf = io.BytesIO()
pickle.dump(model_bundle, pkl_buf)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        label="⬇️ Download predictions (Excel)",
        data=buf.getvalue(),
        file_name=f"{filename.rsplit('.', 1)[0]}_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with d2:
    st.download_button(
        label="⬇️ Download trained model (.pkl)",
        data=pkl_buf.getvalue(),
        file_name=f"{filename.rsplit('.', 1)[0]}_model.pkl",
        mime="application/octet-stream",
        use_container_width=True,
        help="Reusable Python pickle containing the model + preprocessor + metadata.",
    )

# Footer
st.markdown(f"""
<div style='text-align:center; margin-top:1.5rem; color:#9ca3af;
            font-size:.75rem; font-family:Space Mono,monospace;'>
  {best_name} · {task} · {X.shape[1]} features · {len(y):,} samples ·
  test_size={test_size} · cv={cv_folds}
  {'· tuned (' + str(n_iter) + ' iter)' if do_tune else '· default params'}
  {'· class_weight=balanced' if class_weight else ''}
</div>
""", unsafe_allow_html=True)