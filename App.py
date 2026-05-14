# app.py
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
# Global CSS
# ============================================================

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .stApp {
            background: #0b0c10;
        }

        section[data-testid="stSidebar"] {
            background: #0d1117;
            border-right: 1px solid #21262d;
        }

        .app-hero {
            background: linear-gradient(120deg, #0d1117 0%, #111827 55%, #0d1117 100%);
            border: 1px solid #30363d;
            border-left: 4px solid #58a6ff;
            border-radius: 12px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
        }

        .app-hero h1 {
            color: #e6edf3;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0 0 0.3rem 0;
        }

        .app-hero p {
            color: #8b949e;
            margin: 0;
            font-size: 0.95rem;
        }

        .app-badge {
            display: inline-block;
            background: rgba(88, 166, 255, 0.12);
            border: 1px solid rgba(88, 166, 255, 0.3);
            color: #58a6ff;
            font-family: 'Space Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 0.16rem 0.55rem;
            border-radius: 999px;
            margin-right: 0.35rem;
            letter-spacing: 0.08em;
        }

        .app-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.75rem;
        }

        .app-card-title {
            color: #e6edf3;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .app-card-text {
            color: #8b949e;
            font-size: 0.86rem;
            line-height: 1.55;
        }

        .app-section {
            color: #8b949e;
            font-family: 'Space Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 1px solid #21262d;
            padding-bottom: 0.5rem;
            margin: 1.4rem 0 0.9rem 0;
        }

        .status-ok {
            color: #3fb950;
            font-weight: 700;
        }

        .status-missing {
            color: #d29922;
            font-weight: 700;
        }

        .status-bad {
            color: #f85149;
            font-weight: 700;
        }

        .small-muted {
            color: #8b949e;
            font-size: 0.8rem;
            font-family: 'Space Mono', monospace;
        }

        div[data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 0.85rem;
        }

        hr {
            border-color: #21262d;
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
    X_test = get_state_value("X_test")
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
        <div class="app-card">
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
                <span class="app-badge">EXPLORE</span>
                <span class="app-badge">ENGINEER</span>
                <span class="app-badge">TRAIN</span>
                <span class="app-badge">PREDICT</span>
                <span class="app-badge">EXPLAIN</span>
                &nbsp; A no-code machine-learning workspace built with Streamlit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_model_summary()

    st.markdown('<div class="app-section">Workflow</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-title">1. Explore data</div>
                <div class="app-card-text">
                    Upload a CSV or Excel file, inspect missing values, outliers,
                    column types, and correlations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-title">2. Engineer features</div>
                <div class="app-card-text">
                    Impute missing values, encode categoricals, handle outliers,
                    create interactions, and export a transformed dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="app-card">
                <div class="app-card-title">3. Train and explain</div>
                <div class="app-card-text">
                    Train models, benchmark algorithms, run batch predictions,
                    simulate scenarios, and interpret model behavior.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("Start with **Train Model** if you already have a clean dataset.")


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
            <div class="app-card">
                <div class="app-card-title">No dataset in session</div>
                <div class="app-card-text">
                    Upload a dataset in the Data Explorer page or train a model first.
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

with st.sidebar:
    st.markdown("## 🤖 ML AutoPilot")
    st.caption("No-code machine-learning workspace")

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Train Model",
            "Data Explorer",
            "Feature Engineering",
            "Batch Predictor",
            "Model Explainability",
            "What-If Simulator",
        ],
        index=0,
    )

    st.markdown("---")

    if has_model():
        st.markdown('<span class="status-ok">● Model trained</span>', unsafe_allow_html=True)
        st.caption(f"Target: {st.session_state.get('target', '—')}")
        st.caption(f"Task: {st.session_state.get('task', '—')}")
    else:
        st.markdown('<span class="status-missing">● No model yet</span>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Clear session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown(
        """
        <div class="small-muted">
        Streamlit · scikit-learn · pandas · SHAP
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