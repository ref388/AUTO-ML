"""
ML AutoPilot — Streamlit Web App
Upload an Excel file, pick a target column, benchmark algorithms,
optionally tune hyperparameters via RandomizedSearchCV, view parity plots.
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, classification_report
)

# Regressors
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML AutoPilot",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

/* ── background ── */
.stApp {
    background: #0b0c10;
}

/* ── title block ── */
.title-block {
    background: linear-gradient(120deg, #0d1117 0%, #111827 60%, #0d1117 100%);
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 10px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.title-block::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 300px; height: 100%;
    background: radial-gradient(ellipse at right, rgba(88,166,255,0.06) 0%, transparent 70%);
}
.title-block h1 {
    color: #e6edf3;
    font-size: 2.1rem; font-weight: 800;
    letter-spacing: -1px; margin: 0 0 0.25rem 0;
}
.title-block p { color: #8b949e; margin: 0; font-size: 0.92rem; }
.title-block .badge {
    display: inline-block;
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.3);
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; font-weight: 700;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    margin-right: 0.4rem;
    letter-spacing: 1px;
}

/* ── metric cards ── */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1; min-width: 120px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .m-label {
    color: #8b949e;
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.35rem;
}
.metric-card .m-value {
    color: #e6edf3;
    font-size: 1.4rem; font-weight: 700;
    font-family: 'Space Mono', monospace;
}
.metric-card .m-value.blue  { color: #58a6ff; }
.metric-card .m-value.green { color: #3fb950; }
.metric-card .m-value.amber { color: #d29922; }
.metric-card .m-value.red   { color: #f85149; }

/* ── section headers ── */
.section-header {
    color: #8b949e;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── leaderboard ── */
.lb-row {
    display: flex; align-items: center;
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem; gap: 1rem;
    transition: border-color 0.2s;
}
.lb-row.best { border-color: #58a6ff; background: #0d1f38; }
.lb-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem; font-weight: 700;
    color: #30363d; width: 24px; flex-shrink: 0; text-align: center;
}
.lb-rank.gold { color: #d29922; }
.lb-name { color: #e6edf3; font-weight: 600; font-size: 0.95rem; flex: 1; }
.lb-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem; color: #58a6ff; font-weight: 700;
}
.lb-bar-wrap { width: 120px; background: #21262d; border-radius: 4px; height: 6px; }
.lb-bar { height: 6px; border-radius: 4px; background: #58a6ff; }

/* ── info box ── */
.info-box {
    background: rgba(88,166,255,0.07);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    color: #8b949e;
    font-size: 0.85rem;
    margin: 0.8rem 0;
}
.info-box strong { color: #58a6ff; }

/* ── success / warning ── */
.tag-positive {
    background: rgba(63,185,80,0.12); border: 1px solid rgba(63,185,80,0.3);
    color: #3fb950; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.8rem; font-weight: 700;
    font-family: 'Space Mono', monospace;
}
.tag-warning {
    background: rgba(210,153,34,0.12); border: 1px solid rgba(210,153,34,0.3);
    color: #d29922; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.8rem; font-weight: 700;
    font-family: 'Space Mono', monospace;
}

/* matplotlib figure bg */
div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG   = "#0b0c10"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
AMBER     = "#d29922"
RED       = "#f85149"
TEXT      = "#e6edf3"
SUBTEXT   = "#8b949e"

def mpl_dark():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   SUBTEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.6,
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

def detect_task(y: pd.Series):
    """Return 'regression' or 'classification'."""
    if y.dtype == object or y.dtype.name == "category":
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 15 and n_unique / len(y) < 0.05:
        return "classification"
    return "regression"

def preprocess(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop columns that are >60% NaN
    X = X.loc[:, X.isnull().mean() < 0.6]

    # Encode categoricals
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill remaining NaNs
    X = X.fillna(X.median(numeric_only=True))

    task = detect_task(y)
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    else:
        y = y.astype(float)

    return X, y, task


REGRESSORS = {
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Extra Trees":          ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Ridge":                Ridge(),
    "Lasso":                Lasso(max_iter=5000),
    "ElasticNet":           ElasticNet(max_iter=5000),
    "SVR":                  SVR(),
    "KNN":                  KNeighborsRegressor(),
}

CLASSIFIERS = {
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Extra Trees":          ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=2000, random_state=42),
    "SVC":                  SVC(probability=True, random_state=42),
    "KNN":                  KNeighborsClassifier(),
}

PARAM_GRIDS = {
    # Regression
    "Random Forest": {
        "model__n_estimators":      [50, 100, 200, 400],
        "model__max_depth":         [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":      ["sqrt", "log2", 0.5, 0.8],
    },
    "Gradient Boosting": {
        "model__n_estimators":  [50, 100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth":     [2, 3, 4, 6],
        "model__subsample":     [0.6, 0.8, 1.0],
        "model__min_samples_split": [2, 5, 10],
    },
    "Extra Trees": {
        "model__n_estimators":      [50, 100, 200],
        "model__max_depth":         [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features":      ["sqrt", "log2", 0.5],
    },
    "Ridge":       {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    "Lasso":       {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet":  {"model__alpha": [0.001, 0.01, 0.1, 1, 10], "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "SVR":         {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", "auto"], "model__epsilon": [0.01, 0.1, 0.5]},
    "KNN":         {"model__n_neighbors": [3, 5, 7, 10, 15], "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100], "model__penalty": ["l2"], "model__solver": ["lbfgs", "saga"]},
    "SVC":         {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", "auto"], "model__kernel": ["rbf", "linear"]},
}


def make_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def benchmark(X_train, y_train, task, cv=5):
    models = REGRESSORS if task == "regression" else CLASSIFIERS
    scoring = "r2" if task == "regression" else "f1_weighted"
    results = {}
    for name, mdl in models.items():
        pipe = make_pipeline(mdl)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    return results


def tune(X_train, y_train, best_name, task, n_iter=40, cv=5):
    models = REGRESSORS if task == "regression" else CLASSIFIERS
    scoring = "r2" if task == "regression" else "f1_weighted"
    mdl  = models[best_name]
    pipe = make_pipeline(mdl)
    grid = PARAM_GRIDS.get(best_name, {})
    if not grid:
        return pipe.fit(X_train, y_train), {}
    rs = RandomizedSearchCV(
        pipe, grid, n_iter=n_iter, cv=cv,
        scoring=scoring, random_state=42,
        n_jobs=-1, return_train_score=True
    )
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params


def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE %": mape}


def classification_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            pass
    return {"Accuracy": acc, "F1 (weighted)": f1, "ROC-AUC": auc}


# ── Plotting ──────────────────────────────────────────────────────────────────

def parity_plot(y_train, yp_train, y_test, yp_test, target_name):
    mpl_dark()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    for ax, (yt, yp, label, colour) in zip(axes, [
        (y_train, yp_train, "Train", BLUE),
        (y_test,  yp_test,  "Test",  GREEN),
    ]):
        mn = min(yt.min(), yp.min())
        mx = max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "--", color=AMBER, lw=1.5, alpha=0.8, label="Ideal")
        ax.scatter(yt, yp, alpha=0.55, s=20, color=colour, edgecolors="none", label=label)
        r2 = r2_score(yt, yp)
        ax.set_title(f"{label}  ·  R² = {r2:.4f}", fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlabel(f"Actual {target_name}", fontsize=9)
        ax.set_ylabel(f"Predicted {target_name}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Parity Plot — Predicted vs Actual", fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    return fig


def residual_plot(y_train, yp_train, y_test, yp_test):
    mpl_dark()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(DARK_BG)

    for ax, (yt, yp, label, colour) in zip(axes, [
        (y_train, yp_train, "Train", BLUE),
        (y_test,  yp_test,  "Test",  GREEN),
    ]):
        res = yt - yp
        ax.axhline(0, color=AMBER, lw=1.5, linestyle="--", alpha=0.8)
        ax.scatter(yp, res, alpha=0.45, s=18, color=colour, edgecolors="none")
        ax.set_title(f"{label} Residuals", fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual Plot", fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    return fig


def clf_parity_plot(y_train, yp_train, y_test, yp_test, target_name):
    """Actual vs Predicted scatter for classification (jittered index plot)."""
    mpl_dark()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    for ax, (yt, yp, label, colour) in zip(axes, [
        (y_train, yp_train, "Train", BLUE),
        (y_test,  yp_test,  "Test",  GREEN),
    ]):
        yt = np.array(yt); yp = np.array(yp)
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(yt))
        match = yt == yp
        ax.scatter(yt[match]  + jitter[match],  yp[match],  alpha=0.5, s=18, color=colour,   label="Correct",   edgecolors="none")
        ax.scatter(yt[~match] + jitter[~match], yp[~match], alpha=0.6, s=22, color=RED, label="Incorrect", edgecolors="none", marker="x")
        classes = np.unique(np.concatenate([yt, yp]))
        ax.plot(classes, classes, "--", color=AMBER, lw=1.5, alpha=0.8, label="Ideal")
        acc = accuracy_score(yt, yp)
        ax.set_title(f"{label}  ·  Acc = {acc:.4f}", fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlabel(f"Actual {target_name}", fontsize=9)
        ax.set_ylabel(f"Predicted {target_name}", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs Actual (Classification)", fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    return fig


def leaderboard_html(results, task, best_name):
    metric_label = "CV R²" if task == "regression" else "CV F1"
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)
    best_score = sorted_results[0][1]["mean"]
    html = ""
    for rank, (name, info) in enumerate(sorted_results, 1):
        pct = max(0, info["mean"] / (best_score + 1e-9)) * 100
        rank_cls = "gold" if rank == 1 else ""
        row_cls  = "best" if name == best_name else ""
        rank_sym = "🥇" if rank == 1 else str(rank)
        html += f"""
        <div class="lb-row {row_cls}">
            <div class="lb-rank {rank_cls}">{rank_sym}</div>
            <div class="lb-name">{name}</div>
            <div style="flex:1">
                <div class="lb-bar-wrap"><div class="lb-bar" style="width:{pct:.0f}%"></div></div>
            </div>
            <div class="lb-score">{info['mean']:.4f} ± {info['std']:.4f}</div>
        </div>"""
    return f"<div class='section-header'>{metric_label} Leaderboard</div>" + html


def metric_cards_html(metrics):
    colour_map = {"R²": "blue", "Accuracy": "blue", "F1 (weighted)": "green",
                  "ROC-AUC": "green", "MAE": "amber", "RMSE": "amber", "MAPE %": "red"}
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
        <div class="metric-card">
            <div class="m-label">{k}</div>
            <div class="m-value {colour}">{fmt}</div>
        </div>"""
    return f"<div class='metric-row'>{cards}</div>"


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="title-block">
  <h1>🤖 ML AutoPilot</h1>
  <p>
    <span class="badge">UPLOAD</span>
    <span class="badge">BENCHMARK</span>
    <span class="badge">TUNE</span>
    <span class="badge">EVALUATE</span>
    &nbsp; Upload your dataset · pick a target · get the best model automatically
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05,
                          help="Fraction of data held out for testing")
    cv_folds  = st.slider("Cross-validation folds", 3, 10, 5,
                          help="Number of CV folds during benchmarking")

    st.markdown("---")
    st.markdown("### 🔧 Hyperparameter Tuning")
    do_tune   = st.checkbox("Enable RandomizedSearchCV tuning", value=False)
    n_iter    = st.slider("Random search iterations", 10, 100, 40, 10,
                          help="More iterations = better tuning but slower",
                          disabled=not do_tune)

    st.markdown("---")
    st.markdown("### 📊 Display")
    show_residuals = st.checkbox("Show residual plot (regression only)", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 0.72rem;
                color: #30363d; line-height: 1.9;'>
    sklearn · RandomizedSearchCV<br>
    StandardScaler pipeline<br>
    Auto task detection<br>
    Cross-validated benchmark
    </div>
    """, unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your dataset (.xlsx or .csv)",
    type=["xlsx", "xls", "csv"],
    help="Excel or CSV file. First row should be headers."
)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #30363d;
                border: 1px dashed #21262d; border-radius: 10px; margin-top:1rem;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>📂</div>
        <div style='font-size:1.05rem; color:#8b949e;'>Drop an Excel or CSV file to begin</div>
        <div style='font-size:0.82rem; margin-top:0.5rem; font-family: Space Mono, monospace;'>
            .xlsx · .xls · .csv
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.markdown(f"""
<div class="info-box">
    <strong>{uploaded.name}</strong> &nbsp;·&nbsp;
    {df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns &nbsp;·&nbsp;
    {df.isnull().sum().sum():,} missing values
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 Preview data (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# ── Target column ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Target Column</div>', unsafe_allow_html=True)
target = st.selectbox(
    "Select the column you want to predict",
    options=df.columns.tolist(),
    index=len(df.columns) - 1,
    help="This is the output variable your model will learn to predict."
)

# ── Validate & preprocess ─────────────────────────────────────────────────────
if df[target].isnull().mean() > 0.3:
    st.error("Target column has >30% missing values. Please clean your data.")
    st.stop()

df = df.dropna(subset=[target])

try:
    X, y, task = preprocess(df, target)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

task_colour = "green" if task == "regression" else "blue"
st.markdown(f"""
<div class="info-box">
    Detected task: &nbsp;
    <span class="tag-{'positive' if task == 'regression' else 'warning'}">{task.upper()}</span>
    &nbsp;·&nbsp; {X.shape[1]} features after encoding &nbsp;·&nbsp; {len(y):,} samples
</div>
""", unsafe_allow_html=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42,
    stratify=(y if task == "classification" and np.bincount(y).min() >= 2 else None)
)

# ── Run ───────────────────────────────────────────────────────────────────────
run_btn = st.button("🚀 Run AutoPilot", use_container_width=True, type="primary")

if not run_btn:
    st.markdown("""
    <div style='text-align:center; padding: 2rem; color: #30363d;
                font-family: Space Mono, monospace; font-size: 0.8rem;'>
        Configure settings in the sidebar, select your target column, then click Run AutoPilot ↑
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Benchmark ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Benchmarking algorithms…"):
    results = benchmark(X_train, y_train, task, cv=cv_folds)

best_name = max(results, key=lambda k: results[k]["mean"])

st.markdown(leaderboard_html(results, task, best_name), unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
    Best algorithm: <strong>{best_name}</strong> &nbsp;·&nbsp;
    CV score = <strong>{results[best_name]['mean']:.4f}</strong>
    {' &nbsp;·&nbsp; Running RandomizedSearchCV tuning…' if do_tune else ''}
</div>
""", unsafe_allow_html=True)

# ── Tune ──────────────────────────────────────────────────────────────────────
if do_tune:
    with st.spinner(f"🔧 Tuning {best_name} ({n_iter} iterations)…"):
        best_model, best_params = tune(X_train, y_train, best_name, task, n_iter=n_iter, cv=cv_folds)

    if best_params:
        param_str = " &nbsp;·&nbsp; ".join(
            f"<strong>{k.replace('model__','')}</strong>={v}"
            for k, v in best_params.items()
        )
        st.markdown(f'<div class="info-box">Best params: {param_str}</div>', unsafe_allow_html=True)
else:
    models = REGRESSORS if task == "regression" else CLASSIFIERS
    best_model = make_pipeline(models[best_name])
    best_model.fit(X_train, y_train)

st.session_state.model = best_model
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_test = y_test
st.session_state.task = task
st.session_state.target = target
st.session_state.feature_names = X.columns.tolist()
st.session_state.df = df.copy()

# ── Evaluate ──────────────────────────────────────────────────────────────────
yp_train = best_model.predict(X_train)
yp_test  = best_model.predict(X_test)

st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)

col_tr, col_te = st.columns(2)

if task == "regression":
    y_train_arr = np.array(y_train, dtype=float)
    y_test_arr  = np.array(y_test,  dtype=float)

    with col_tr:
        st.markdown("**Train**")
        st.markdown(metric_cards_html(regression_metrics(y_train_arr, yp_train)), unsafe_allow_html=True)
    with col_te:
        st.markdown("**Test**")
        st.markdown(metric_cards_html(regression_metrics(y_test_arr, yp_test)), unsafe_allow_html=True)

    # Parity plot
    st.markdown('<div class="section-header">Parity Plot</div>', unsafe_allow_html=True)
    fig = parity_plot(y_train_arr, yp_train, y_test_arr, yp_test, target)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Residual plot
    if show_residuals:
        st.markdown('<div class="section-header">Residual Plot</div>', unsafe_allow_html=True)
        fig2 = residual_plot(y_train_arr, yp_train, y_test_arr, yp_test)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

else:
    y_prob_train = best_model.predict_proba(X_train) if hasattr(best_model, "predict_proba") else None
    y_prob_test  = best_model.predict_proba(X_test)  if hasattr(best_model, "predict_proba") else None

    with col_tr:
        st.markdown("**Train**")
        st.markdown(metric_cards_html(classification_metrics(y_train, yp_train, y_prob_train)), unsafe_allow_html=True)
    with col_te:
        st.markdown("**Test**")
        st.markdown(metric_cards_html(classification_metrics(y_test, yp_test, y_prob_test)), unsafe_allow_html=True)

    # Parity plot
    st.markdown('<div class="section-header">Predicted vs Actual</div>', unsafe_allow_html=True)
    fig = clf_parity_plot(np.array(y_train), yp_train, np.array(y_test), yp_test, target)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Classification report
    with st.expander("📋 Full classification report (test set)"):
        report = classification_report(y_test, yp_test, output_dict=False)
        st.code(report, language="text")

# ── Download predictions ───────────────────────────────────────────────────────
st.markdown("---")

pred_df = pd.DataFrame({
    "Actual":    np.concatenate([np.array(y_train), np.array(y_test)]),
    "Predicted": np.concatenate([yp_train, yp_test]),
    "Split":     ["Train"] * len(y_train) + ["Test"] * len(y_test),
})
buf = io.BytesIO()
pred_df.to_excel(buf, index=False)
st.download_button(
    label="⬇️ Download predictions (Excel)",
    data=buf.getvalue(),
    file_name=f"{uploaded.name.rsplit('.',1)[0]}_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

st.markdown(f"""
<div style='text-align:center; margin-top:1.5rem; color:#21262d;
            font-size:0.75rem; font-family: Space Mono, monospace;'>
  {best_name} · {task} · {X.shape[1]} features · {len(y):,} samples ·
  test_size={test_size} · cv={cv_folds}
  {'· tuned (' + str(n_iter) + ' iter)' if do_tune else '· default params'}
</div>
""", unsafe_allow_html=True)