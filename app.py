# ============================================================
# ML AutoPilot — Main Streamlit Front End
# Upload this file at the root of your GitHub repo.
#
# Expected files in the same folder:
#   app.py
#   automl_page.py
#   data_explorer.py
#   feature_engineering.py
#   batch_predictor.py
#   model_explainability.py
#   what_if_simulator.py
# ============================================================

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="ML AutoPilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Safe imports for your widget modules
# ============================================================

try:
    from data_explorer import render_data_explorer
except Exception as e:
    render_data_explorer = None
    DATA_EXPLORER_IMPORT_ERROR = e
else:
    DATA_EXPLORER_IMPORT_ERROR = None

try:
    from feature_engineering import render_feature_engineering
except Exception as e:
    render_feature_engineering = None
    FEATURE_ENGINEERING_IMPORT_ERROR = e
else:
    FEATURE_ENGINEERING_IMPORT_ERROR = None

try:
    from batch_predictor import render_batch_predictor
except Exception as e:
    render_batch_predictor = None
    BATCH_PREDICTOR_IMPORT_ERROR = e
else:
    BATCH_PREDICTOR_IMPORT_ERROR = None

try:
    from model_explainability import render_explainability
except Exception as e:
    render_explainability = None
    EXPLAINABILITY_IMPORT_ERROR = e
else:
    EXPLAINABILITY_IMPORT_ERROR = None

try:
    from what_if_simulator import render_what_if
except Exception as e:
    render_what_if = None
    WHAT_IF_IMPORT_ERROR = e
else:
    WHAT_IF_IMPORT_ERROR = None


# ============================================================
# Global CSS — light/airy theme (no pure-black background)
# ============================================================

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* ── Base palette ───────────────────────────────────────
           bg       #f6f7fb  light grey-blue
           surface  #ffffff  white panels
           border   #e4e7ec  soft grey
           text     #1f2937  near-black
           sub      #6b7280  muted
           primary  #4f46e5  indigo
           accent   #14b8a6  teal
           amber    #f59e0b
           red      #ef4444
           green    #10b981
        ───────────────────────────────────────────────────────── */

        html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #1f2937; }

        .stApp {
            background:
              radial-gradient(1200px 600px at 10% -10%, rgba(79,70,229,0.06) 0%, transparent 60%),
              radial-gradient(1000px 500px at 100% 0%, rgba(20,184,166,0.05) 0%, transparent 55%),
              linear-gradient(180deg, #f6f7fb 0%, #eef0f6 100%);
        }

        /* main block container — narrower, with breathing room */
        .block-container {
            max-width: 1280px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* ── Sidebar ─────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e4e7ec;
        }
        section[data-testid="stSidebar"] * { color: #1f2937; }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #6b7280;
        }

        /* radio nav restyle */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.25rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 0.55rem 0.7rem;
            transition: all 0.15s ease;
            cursor: pointer;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: #f3f4f6;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: rgba(79,70,229,0.08);
            border-color: rgba(79,70,229,0.25);
        }

        /* ── Hero ────────────────────────────────────────────── */
        .app-hero {
            background:
              linear-gradient(135deg, #ffffff 0%, #fafbff 100%);
            border: 1px solid #e4e7ec;
            border-radius: 16px;
            padding: 1.8rem 2.2rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 4px 12px rgba(16,24,40,0.04);
        }
        .app-hero::before {
            content: '';
            position: absolute; top: -40%; right: -10%;
            width: 420px; height: 420px;
            background: radial-gradient(circle, rgba(79,70,229,0.12) 0%, transparent 60%);
            pointer-events: none;
        }
        .app-hero::after {
            content: '';
            position: absolute; bottom: -50%; left: 40%;
            width: 380px; height: 380px;
            background: radial-gradient(circle, rgba(20,184,166,0.10) 0%, transparent 60%);
            pointer-events: none;
        }
        .app-hero h1 {
            color: #1f2937;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0 0 0.4rem 0;
            position: relative; z-index: 1;
        }
        .app-hero p {
            color: #6b7280;
            margin: 0;
            font-size: 0.98rem;
            position: relative; z-index: 1;
        }

        /* ── Badges ──────────────────────────────────────────── */
        .app-badge {
            display: inline-block;
            background: rgba(79,70,229,0.08);
            border: 1px solid rgba(79,70,229,0.18);
            color: #4f46e5;
            font-family: 'Space Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            margin-right: 0.35rem;
            letter-spacing: 0.08em;
        }
        .app-badge.teal { background: rgba(20,184,166,0.08); border-color: rgba(20,184,166,0.2); color: #0d9488; }
        .app-badge.amber{ background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.2); color: #b45309; }

        /* ── Cards ───────────────────────────────────────────── */
        .app-card {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 1px 2px rgba(16,24,40,0.03);
            height: 100%;
        }
        .app-card .app-card-icon {
            display: inline-flex;
            align-items: center; justify-content: center;
            width: 40px; height: 40px;
            background: rgba(79,70,229,0.08);
            border-radius: 10px;
            font-size: 1.3rem;
            margin-bottom: 0.7rem;
        }
        .app-card .app-card-icon.teal  { background: rgba(20,184,166,0.10); }
        .app-card .app-card-icon.amber { background: rgba(245,158,11,0.10); }
        .app-card .app-card-icon.green { background: rgba(16,185,129,0.10); }
        .app-card .app-card-icon.pink  { background: rgba(236,72,153,0.10); }
        .app-card .app-card-icon.cyan  { background: rgba(6,182,212,0.10); }

        .app-card-title {
            color: #1f2937;
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .app-card-text {
            color: #6b7280;
            font-size: 0.88rem;
            line-height: 1.6;
        }
        .app-card-step {
            color: #4f46e5;
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.3rem;
            display: block;
        }

        /* ── Section header ──────────────────────────────────── */
        .app-section {
            color: #6b7280;
            font-family: 'Space Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            border-bottom: 1px solid #e4e7ec;
            padding-bottom: 0.5rem;
            margin: 1.6rem 0 1rem 0;
        }

        /* ── Status pills ───────────────────────────────────── */
        .status-pill {
            display: inline-flex; align-items: center;
            gap: 0.4rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
        }
        .status-pill::before {
            content: ''; width: 6px; height: 6px; border-radius: 50%;
        }
        .status-pill.ok     { background: rgba(16,185,129,0.10); color: #047857; }
        .status-pill.ok::before     { background: #10b981; }
        .status-pill.warn   { background: rgba(245,158,11,0.10); color: #b45309; }
        .status-pill.warn::before   { background: #f59e0b; }
        .status-pill.bad    { background: rgba(239,68,68,0.10);  color: #b91c1c; }
        .status-pill.bad::before    { background: #ef4444; }

        .small-muted {
            color: #6b7280;
            font-size: 0.78rem;
            font-family: 'Space Mono', monospace;
        }

        /* ── Metric cards (Streamlit) ────────────────────────── */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 1px 2px rgba(16,24,40,0.03);
        }
        div[data-testid="stMetric"] label { color: #6b7280 !important; font-size: 0.75rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
        div[data-testid="stMetricValue"] { color: #1f2937 !important; font-family: 'Space Mono', monospace; font-weight: 700; }

        /* ── Buttons ─────────────────────────────────────────── */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #e4e7ec;
            background: #ffffff;
            color: #1f2937;
            font-weight: 600;
            padding: 0.45rem 1rem;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            border-color: #c7d2fe;
            background: #f5f5ff;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: #ffffff;
            border: none;
            box-shadow: 0 1px 2px rgba(79,70,229,0.2), 0 4px 12px rgba(79,70,229,0.18);
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 1px 2px rgba(79,70,229,0.25), 0 6px 16px rgba(79,70,229,0.25);
            transform: translateY(-1px);
        }

        /* ── Forms / inputs ──────────────────────────────────── */
        .stTextInput input, .stTextArea textarea, .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] {
            background: #ffffff;
            border-radius: 8px;
        }

        /* ── Tabs ────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.2rem;
            border-bottom: 1px solid #e4e7ec;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
            color: #6b7280;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            color: #4f46e5 !important;
            background: rgba(79,70,229,0.06) !important;
        }

        /* ── Misc ────────────────────────────────────────────── */
        hr { border-color: #e4e7ec; }

        /* file uploader */
        [data-testid="stFileUploader"] section {
            background: #ffffff;
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            padding: 1.2rem;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #4f46e5;
            background: #fafbff;
        }

        /* expander */
        details[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 10px;
        }

        /* code blocks */
        code, pre {
            background: #f3f4f6 !important;
            color: #1f2937 !important;
            border-radius: 6px;
        }

        /* dataframes */
        [data-testid="stDataFrame"] {
            background: #ffffff;
            border-radius: 10px;
            border: 1px solid #e4e7ec;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()


# ============================================================
# Helpers
# ============================================================

def has_model() -> bool:
    required = ["model", "X_train", "task", "target", "feature_names"]
    return all(key in st.session_state for key in required)


def get_state_value(key: str, default: Any = None) -> Any:
    return st.session_state[key] if key in st.session_state else default


def session_model_summary() -> None:
    st.markdown('<div class="app-section">Session status</div>', unsafe_allow_html=True)

    model = get_state_value("model")
    X_train = get_state_value("X_train")
    task = get_state_value("task")
    target = get_state_value("target")
    feature_names = get_state_value("feature_names")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if model is None:
            st.metric("Model", "Not trained")
        else:
            if hasattr(model, "named_steps"):
                final_step = list(model.named_steps.values())[-1]
                model_name = type(final_step).__name__
            else:
                model_name = type(model).__name__
            st.metric("Model", model_name)

    with c2:
        st.metric("Task", str(task).title() if task else "—")

    with c3:
        if isinstance(X_train, pd.DataFrame):
            st.metric("Train rows", f"{len(X_train):,}")
        else:
            st.metric("Train rows", "—")

    with c4:
        st.metric("Target", target if target else "—")

    if feature_names:
        st.caption(f"{len(feature_names)} active feature(s).")


def require_trained_model(page_name: str) -> bool:
    if has_model():
        return True

    st.markdown(
        f"""
        <div class="app-card" style="border-left: 4px solid #f59e0b;">
            <div class="app-card-title">No trained model available</div>
            <div class="app-card-text">
                The <strong>{page_name}</strong> page needs a trained model from the AutoPilot page.
                Go to <strong>Train Model</strong>, upload a dataset, select a target, and run the model first.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return False


def render_import_error(module_name: str, error: Exception | None) -> None:
    st.error(
        f"Could not import `{module_name}`. "
        f"Check the filename, function name, and dependencies. Error: {error}"
    )


def render_home() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <h1>🤖 ML AutoPilot</h1>
            <p>
                <span class="app-badge">UPLOAD</span>
                <span class="app-badge teal">EXPLORE</span>
                <span class="app-badge amber">ENGINEER</span>
                <span class="app-badge">TRAIN</span>
                <span class="app-badge teal">PREDICT</span>
                <span class="app-badge amber">EXPLAIN</span>
                <br><br>
                A no-code machine-learning workspace. Bring your data, get a trained
                model, predictions, and explanations &mdash; all without writing a line of code.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_model_summary()

    # ── Workflow cards ──────────────────────────────────────
    st.markdown('<div class="app-section">Workflow</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-icon">🔍</div>
                <span class="app-card-step">Step 1</span>
                <div class="app-card-title">Explore your data</div>
                <div class="app-card-text">
                    Upload a CSV or Excel file. Inspect missing values, outliers,
                    column types, and correlations &mdash; with smart suggestions
                    to fix common issues.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-icon teal">🔧</div>
                <span class="app-card-step">Step 2</span>
                <div class="app-card-title">Engineer features</div>
                <div class="app-card-text">
                    Impute missing values, encode categoricals, handle outliers,
                    create interactions. Every step is recorded so you can undo or
                    replay your pipeline later.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-icon amber">🤖</div>
                <span class="app-card-step">Step 3</span>
                <div class="app-card-title">Train your model</div>
                <div class="app-card-text">
                    Benchmark 8 algorithms automatically. Optionally tune
                    hyperparameters. Get parity plots, R&sup2;, accuracy, and
                    everything you need to trust the result.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-icon green">📦</div>
                <span class="app-card-step">Step 4</span>
                <div class="app-card-title">Batch predict</div>
                <div class="app-card-text">
                    Upload new data next month. The model predicts in seconds,
                    flags low-confidence rows, and exports an enriched Excel file.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-icon pink">🎯</div>
                <span class="app-card-step">Step 5</span>
                <div class="app-card-title">Run what-if scenarios</div>
                <div class="app-card-text">
                    Slide a feature, see the prediction change instantly.
                    Save and compare scenarios &mdash; ideal for stakeholder demos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c6:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-icon cyan">🔬</div>
                <span class="app-card-step">Step 6</span>
                <div class="app-card-title">Explain decisions</div>
                <div class="app-card-text">
                    SHAP-powered explanations show which features matter, how they
                    push predictions up or down, and why any single row got its result.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Getting started ──────────────────────────────────────
    st.markdown('<div class="app-section">Getting started</div>', unsafe_allow_html=True)

    if has_model():
        st.markdown(
            f"""
            <div class="app-card" style="border-left: 4px solid #10b981;">
                <div class="app-card-title">✅ A model is loaded and ready</div>
                <div class="app-card-text">
                    You can now use any of the post-training pages: <strong>Batch Predictor</strong>,
                    <strong>What-If Simulator</strong>, or <strong>Model Explainability</strong>.
                    Use the sidebar to navigate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="app-card" style="border-left: 4px solid #4f46e5;">
                <div class="app-card-title">👉 Start here</div>
                <div class="app-card-text">
                    Head to <strong>Train Model</strong> in the sidebar, upload a CSV or Excel file,
                    pick the column you want to predict, and click Run. The whole pipeline takes
                    about a minute for most datasets.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_train_model() -> None:
    """
    Runs your AutoML page.

    Important:
    - `automl_page.py` should be in the same folder as this `app.py`.
    - Remove/comment `st.set_page_config(...)` from `automl_page.py`,
      because this root app already calls it.
    - Make sure your AutoML page saves the trained objects into session_state:
        st.session_state.model
        st.session_state.X_train
        st.session_state.X_test
        st.session_state.y_test
        st.session_state.task
        st.session_state.target
        st.session_state.feature_names
        st.session_state.df
    """
    automl_path = Path(__file__).parent / "automl_page.py"

    if not automl_path.exists():
        st.error(
            "Could not find `automl_page.py`. "
            "Rename `AutoML.py` to `automl_page.py` and place it next to `app.py`."
        )
        return

    try:
        runpy.run_path(str(automl_path), run_name="__main__")
    except Exception as e:
        st.error("The AutoML training page failed to run.")
        st.exception(e)


def render_data_page() -> None:
    if render_data_explorer is None:
        render_import_error("data_explorer.py", DATA_EXPLORER_IMPORT_ERROR)
        return

    df = get_state_value("df")
    filename = get_state_value("filename", "uploaded_dataset")

    if isinstance(df, pd.DataFrame):
        render_data_explorer(df=df, filename=filename)
    else:
        render_data_explorer()


def render_feature_page() -> None:
    if render_feature_engineering is None:
        render_import_error("feature_engineering.py", FEATURE_ENGINEERING_IMPORT_ERROR)
        return

    df = get_state_value("df")
    filename = get_state_value("filename", "uploaded_dataset")

    if not isinstance(df, pd.DataFrame):
        st.markdown(
            """
            <div class="app-card" style="border-left: 4px solid #f59e0b;">
                <div class="app-card-title">No dataset in session</div>
                <div class="app-card-text">
                    Upload a dataset in the <strong>Data Explorer</strong> page or train a model first.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    render_feature_engineering(df=df, filename=filename)


def render_batch_page() -> None:
    if render_batch_predictor is None:
        render_import_error("batch_predictor.py", BATCH_PREDICTOR_IMPORT_ERROR)
        return

    if not require_trained_model("Batch Predictor"):
        return

    render_batch_predictor(
        model=st.session_state.model,
        X_train=st.session_state.X_train,
        task=st.session_state.task,
        feature_names=st.session_state.feature_names,
        target_name=st.session_state.target,
    )


def render_explainability_page() -> None:
    if render_explainability is None:
        render_import_error("model_explainability.py", EXPLAINABILITY_IMPORT_ERROR)
        return

    if not require_trained_model("Model Explainability"):
        return

    render_explainability(
        model=st.session_state.model,
        X_train=st.session_state.X_train,
        X_test=st.session_state.get("X_test"),
        y_test=st.session_state.get("y_test"),
        task=st.session_state.task,
        feature_names=st.session_state.feature_names,
        target_name=st.session_state.target,
    )


def render_what_if_page() -> None:
    if render_what_if is None:
        render_import_error("what_if_simulator.py", WHAT_IF_IMPORT_ERROR)
        return

    if not require_trained_model("What-If Simulator"):
        return

    render_what_if(
        model=st.session_state.model,
        X_train=st.session_state.X_train,
        task=st.session_state.task,
        feature_names=st.session_state.feature_names,
        target_name=st.session_state.target,
    )


# ============================================================
# Sidebar navigation
# ============================================================

NAV_ITEMS = [
    ("🏠  Home",                "Home"),
    ("🤖  Train Model",         "Train Model"),
    ("🔍  Data Explorer",       "Data Explorer"),
    ("🔧  Feature Engineering", "Feature Engineering"),
    ("📦  Batch Predictor",     "Batch Predictor"),
    ("🔬  Model Explainability","Model Explainability"),
    ("🎯  What-If Simulator",   "What-If Simulator"),
]

with st.sidebar:
    st.markdown(
        """
        <div style='padding: 0.5rem 0 0.2rem 0;'>
            <div style='font-size:1.4rem; font-weight:800; color:#1f2937; letter-spacing:-0.02em;'>
                🤖 ML AutoPilot
            </div>
            <div style='color:#6b7280; font-size:0.85rem; margin-top:0.1rem;'>
                No-code ML workspace
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [item[1] for item in NAV_ITEMS],
        format_func=lambda v: dict((b, a) for a, b in NAV_ITEMS)[v],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Session status pill
    if has_model():
        st.markdown('<span class="status-pill ok">Model ready</span>', unsafe_allow_html=True)
        st.caption(f"Target: **{st.session_state.get('target', '—')}**")
        st.caption(f"Task: {st.session_state.get('task', '—')}")
    else:
        st.markdown('<span class="status-pill warn">No model yet</span>', unsafe_allow_html=True)
        st.caption("Train a model to unlock the post-training pages.")

    st.markdown("---")

    if st.button("🗑️ Clear session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown(
        """
        <div class="small-muted" style="margin-top: 1.2rem; line-height: 1.6;">
            Streamlit · scikit-learn<br>
            pandas · SHAP
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Router
# ============================================================

if page == "Home":
    render_home()

elif page == "Train Model":
    render_train_model()

elif page == "Data Explorer":
    render_data_page()

elif page == "Feature Engineering":
    render_feature_page()

elif page == "Batch Predictor":
    render_batch_page()

elif page == "Model Explainability":
    render_explainability_page()

elif page == "What-If Simulator":
    render_what_if_page()
