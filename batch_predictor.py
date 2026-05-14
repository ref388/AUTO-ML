"""
ML AutoPilot — Batch Predictor widget
Drop-in module for the larger ML AutoPilot app.

Run a trained model against a fresh dataset:
  • Upload new data (Excel or CSV)
  • Schema validation against training data
  • Auto-align columns (handle missing/extra columns, dtype mismatches)
  • Run predictions in batches with a progress bar
  • Per-row confidence score (probability for classification, prediction-interval
    width estimate for regression)
  • Distribution visualization of predictions
  • Filter / sort / inspect predictions interactively
  • Download enriched dataset (original data + predictions + confidence)

USAGE
---------------------------
    from batch_predictor import render_batch_predictor

    if page == "Batch Predictor":
        render_batch_predictor(
            model=st.session_state.model,
            X_train=st.session_state.X_train,
            task=st.session_state.task,
            feature_names=st.session_state.feature_names,
            target_name=st.session_state.target,
        )
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Design tokens (shared) ────────────────────────────────────────────────────
DARK_BG  = "#f6f7fb"
PANEL_BG = "#ffffff"
BORDER   = "#e4e7ec"
BLUE     = "#4f46e5"
GREEN    = "#10b981"
AMBER    = "#f59e0b"
RED      = "#ef4444"
PURPLE   = "#8b5cf6"
TEXT     = "#1f2937"
SUBTEXT  = "#6b7280"


def _inject_css_once():
    if st.session_state.get("_bp_css_done"):
        return
    st.markdown("""
    <style>
    .bp-section {
        color:#6b7280; font-family:'Space Mono',monospace;
        font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:2px;
        border-bottom:1px solid #f3f4f6; padding-bottom:.5rem; margin:1.8rem 0 1rem;
    }
    .bp-stat-row { display:flex; gap:.8rem; flex-wrap:wrap; margin:1rem 0; }
    .bp-stat-card {
        flex:1; min-width:130px;
        background:#ffffff; border:1px solid #e4e7ec; border-radius:8px;
        padding:1rem 1.2rem; text-align:center;
    }
    .bp-stat-card .s-label {
        color:#6b7280; font-size:.68rem; font-weight:600; text-transform:uppercase;
        letter-spacing:1.5px; font-family:'Space Mono',monospace; margin-bottom:.35rem;
    }
    .bp-stat-card .s-value { color:#1f2937; font-size:1.4rem; font-weight:700; font-family:'Space Mono',monospace; }
    .bp-stat-card .s-value.blue   { color:#4f46e5; }
    .bp-stat-card .s-value.green  { color:#10b981; }
    .bp-stat-card .s-value.amber  { color:#f59e0b; }
    .bp-stat-card .s-value.red    { color:#ef4444; }
    .bp-stat-card .s-value.purple { color:#8b5cf6; }

    .bp-info {
        background:rgba(79,70,229,.07); border:1px solid rgba(79,70,229,.2);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.6rem 0;
    }
    .bp-info strong { color:#4f46e5; }
    .bp-good {
        background:rgba(16,185,129,.07); border:1px solid rgba(16,185,129,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .bp-good strong { color:#10b981; }
    .bp-warn {
        background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .bp-warn strong { color:#f59e0b; }
    .bp-err {
        background:rgba(239,68,68,.07); border:1px solid rgba(239,68,68,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .bp-err strong { color:#ef4444; }

    /* schema check row */
    .bp-schema-row {
        display:flex; align-items:center;
        background:#ffffff; border:1px solid #f3f4f6;
        border-radius:8px; padding:.55rem .9rem;
        margin-bottom:.35rem; gap:.8rem;
        font-size:.85rem;
    }
    .bp-schema-icon { width:20px; flex-shrink:0; text-align:center; font-size:1rem; }
    .bp-schema-name { color:#1f2937; font-weight:600; flex:1; }
    .bp-schema-note {
        font-family:'Space Mono',monospace; font-size:.78rem;
        color:#6b7280;
    }
    .bp-schema-note.ok    { color:#10b981; }
    .bp-schema-note.warn  { color:#f59e0b; }
    .bp-schema-note.error { color:#ef4444; }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_bp_css_done"] = True


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib theme
# ══════════════════════════════════════════════════════════════════════════════

def _mpl_dark():
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


# ══════════════════════════════════════════════════════════════════════════════
# Schema validation & alignment
# ══════════════════════════════════════════════════════════════════════════════

def _validate_schema(new_df: pd.DataFrame, X_train: pd.DataFrame) -> dict:
    """
    Compare new_df columns against the training schema.
    Returns a dict with: missing, extra, dtype_mismatch, ok columns.
    """
    train_cols = set(X_train.columns)
    new_cols   = set(new_df.columns)

    missing = sorted(train_cols - new_cols)
    extra   = sorted(new_cols - train_cols)
    common  = sorted(train_cols & new_cols)

    dtype_mismatch = []
    for col in common:
        train_dtype = X_train[col].dtype
        new_dtype   = new_df[col].dtype
        train_kind  = train_dtype.kind
        new_kind    = new_dtype.kind
        # We tolerate numeric ↔ numeric, but flag string vs numeric mismatches
        if train_kind in "biufc" and new_kind not in "biufc":
            dtype_mismatch.append({
                "column":      col,
                "expected":    str(train_dtype),
                "got":         str(new_dtype),
            })

    return {
        "missing":         missing,
        "extra":           extra,
        "dtype_mismatch":  dtype_mismatch,
        "common":          common,
        "n_rows":          len(new_df),
    }


def _align_to_schema(new_df: pd.DataFrame, X_train: pd.DataFrame,
                     fill_strategy: str = "median") -> pd.DataFrame:
    """
    Align new_df to match X_train's schema:
    - Add missing columns with imputed values
    - Drop extra columns
    - Reorder to match training order
    - Coerce dtypes where possible
    """
    aligned = new_df.copy()

    # Drop extra columns
    extra = [c for c in aligned.columns if c not in X_train.columns]
    if extra:
        aligned = aligned.drop(columns=extra)

    # Add missing columns
    missing = [c for c in X_train.columns if c not in aligned.columns]
    for col in missing:
        s = X_train[col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            fill = s.median() if fill_strategy == "median" else s.mean()
        else:
            fill = s.mode().iloc[0] if not s.mode().empty else ""
        aligned[col] = fill

    # Reorder
    aligned = aligned[X_train.columns]

    # Coerce dtypes
    for col in aligned.columns:
        train_dtype = X_train[col].dtype
        try:
            if pd.api.types.is_numeric_dtype(train_dtype):
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            aligned[col] = aligned[col].astype(train_dtype, errors="ignore")
        except Exception:
            pass

    # Fill any NaNs introduced by coercion
    for col in aligned.columns:
        if aligned[col].isnull().any():
            s = X_train[col].dropna()
            if pd.api.types.is_numeric_dtype(s):
                fill = s.median()
            else:
                fill = s.mode().iloc[0] if not s.mode().empty else ""
            aligned[col] = aligned[col].fillna(fill)

    return aligned


# ══════════════════════════════════════════════════════════════════════════════
# Batched prediction
# ══════════════════════════════════════════════════════════════════════════════

def _batched_predict(model, X: pd.DataFrame, task: str, batch_size: int = 1000,
                     progress_cb=None):
    """
    Run model predictions in batches with optional progress callback.
    Returns (predictions, probabilities_dict_or_None).
    """
    n = len(X)
    preds = []
    probas = [] if task == "classification" and hasattr(model, "predict_proba") else None

    n_batches = -(-n // batch_size)
    for i in range(n_batches):
        start = i * batch_size
        end   = min(start + batch_size, n)
        chunk = X.iloc[start:end]

        preds.append(model.predict(chunk))
        if probas is not None:
            try:
                probas.append(model.predict_proba(chunk))
            except Exception:
                probas = None

        if progress_cb:
            progress_cb((i + 1) / n_batches)

    predictions = np.concatenate(preds)
    probabilities = np.concatenate(probas) if probas else None
    return predictions, probabilities


def _compute_confidence(predictions, probabilities, task: str, model=None, X=None) -> np.ndarray:
    """
    Compute a per-row confidence score in [0, 1].
    Classification: max class probability.
    Regression: 1 / (1 + tree variance), if the estimator is a tree ensemble; else 1.0.
    """
    if task == "classification" and probabilities is not None:
        return probabilities.max(axis=1)

    if task == "regression":
        # Try to extract individual tree predictions for ensemble variance
        try:
            est = model.named_steps[list(model.named_steps.keys())[-1]] if hasattr(model, "named_steps") else model
            # Transform X with the pipeline steps before the final estimator
            if hasattr(model, "named_steps"):
                X_t = X.copy()
                for step in list(model.named_steps.values())[:-1]:
                    X_t = step.transform(X_t)
            else:
                X_t = X.values

            if hasattr(est, "estimators_"):
                # Random Forest / Extra Trees: collect per-tree predictions
                tree_preds = np.array([t.predict(X_t) for t in est.estimators_])
                stds = tree_preds.std(axis=0)
                # Normalize std → confidence in [0, 1] using the training-time std as scale
                scale = stds.max() if stds.max() > 0 else 1.0
                return 1 - np.clip(stds / scale, 0, 1)
        except Exception:
            pass

    return np.ones(len(predictions))


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _prediction_dist(predictions, task: str):
    _mpl_dark()
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)

    if task == "classification":
        s = pd.Series(predictions).value_counts().sort_index()
        ax.bar(s.index.astype(str), s.values, color=BLUE, alpha=0.85, edgecolor="none")
        ax.set_xlabel("Predicted class", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Distribution of predictions", fontsize=11, fontweight="bold", color=TEXT)
        for i, (cls, n) in enumerate(s.items()):
            ax.text(i, n, f" {n:,}", ha="center", va="bottom", fontsize=9, color=SUBTEXT)
    else:
        ax.hist(predictions, bins=40, color=BLUE, alpha=0.85, edgecolor="none")
        ax.axvline(np.mean(predictions), color=AMBER, lw=1.5, linestyle="--",
                   label=f"Mean = {np.mean(predictions):.3g}")
        ax.axvline(np.median(predictions), color=GREEN, lw=1.5, linestyle="--",
                   label=f"Median = {np.median(predictions):.3g}")
        ax.set_xlabel("Predicted value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Distribution of predictions", fontsize=11, fontweight="bold", color=TEXT)
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def _confidence_dist(confidence: np.ndarray):
    _mpl_dark()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(DARK_BG)

    ax.hist(confidence, bins=30, color=PURPLE, alpha=0.85, edgecolor="none")
    median_conf = float(np.median(confidence))
    low_thresh  = 0.6
    n_low       = int((confidence < low_thresh).sum())

    ax.axvline(median_conf, color=AMBER, lw=1.5, linestyle="--",
               label=f"Median = {median_conf:.2f}")
    ax.axvline(low_thresh, color=RED, lw=1.5, linestyle=":",
               label=f"Low-confidence threshold = {low_thresh}")

    ax.set_xlabel("Confidence score (0 = uncertain, 1 = very confident)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Confidence distribution — {n_low:,} low-confidence predictions",
                 fontsize=11, fontweight="bold", color=TEXT)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Stat-card helper
# ══════════════════════════════════════════════════════════════════════════════

def _stat_cards(*cards) -> str:
    html = "<div class='bp-stat-row'>"
    for label, value, colour in cards:
        html += f"""
        <div class='bp-stat-card'>
            <div class='s-label'>{label}</div>
            <div class='s-value {colour}'>{value}</div>
        </div>"""
    return html + "</div>"


def _schema_row(icon: str, name: str, note: str, note_class: str = "") -> str:
    return f"""
    <div class='bp-schema-row'>
        <div class='bp-schema-icon'>{icon}</div>
        <div class='bp-schema-name'>{name}</div>
        <div class='bp-schema-note {note_class}'>{note}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_batch_predictor(
    model=None,
    X_train: pd.DataFrame = None,
    task: str = "regression",
    feature_names: list = None,
    target_name: str = "prediction",
):
    """
    Render the Batch Predictor widget.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
    X_train : pandas.DataFrame — used to validate the incoming schema
    task : "regression" | "classification"
    feature_names : list[str] — defaults to X_train.columns
    target_name : str — name used for the output column (e.g. "churn_predicted")
    """
    _inject_css_once()

    st.markdown("## 📦 Batch Predictor")
    st.caption("Upload new data, run your trained model, download enriched predictions. No retraining needed.")

    # ── Guard ─────────────────────────────────────────────────────────────────
    if model is None or X_train is None:
        st.markdown("""
        <div class='bp-warn'>
            ⚠️ <strong>No trained model in session.</strong>
            Run the AutoPilot module first, then come back here to predict on new data.
        </div>
        """, unsafe_allow_html=True)
        return

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    # ── Model summary ─────────────────────────────────────────────────────────
    model_name = type(model.named_steps[list(model.named_steps.keys())[-1]]).__name__ \
                 if hasattr(model, "named_steps") else type(model).__name__

    st.markdown('<div class="bp-section">Active model</div>', unsafe_allow_html=True)
    st.markdown(_stat_cards(
        ("Model",            model_name,              "blue"),
        ("Task",             task.title(),            "purple"),
        ("Features expected",str(X_train.shape[1]),   "blue"),
        ("Predicts",         target_name,             "green"),
    ), unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="bp-section">Upload new data</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='bp-info'>
        Upload a file with the same columns as your training data — minus the target column.
        Don't worry about column order; we'll handle that. Missing or extra columns will be flagged below.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload new dataset (.xlsx, .xls, or .csv)",
        type=["xlsx", "xls", "csv"],
        key="bp_uploader",
    )

    if uploaded is None:
        # Helper: download a blank template
        template = X_train.head(0).copy()
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            template.to_excel(writer, index=False, sheet_name="Template")
        st.download_button(
            label="📄 Download blank input template (Excel)",
            data=buf.getvalue(),
            file_name=f"{target_name}_input_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="bp_template_dl",
            help="An empty file with the exact columns your model expects.",
        )

        st.markdown("""
        <div style='text-align:center;padding:3rem 2rem;color:#e4e7ec;
                    border:1px dashed #f3f4f6;border-radius:10px;margin-top:1rem;'>
            <div style='font-size:2.5rem;margin-bottom:1rem;'>📦</div>
            <div style='font-size:1rem;color:#6b7280;'>Drop your new data to generate predictions</div>
            <div style='font-size:.8rem;margin-top:.5rem;font-family:Space Mono,monospace;color:#e4e7ec'>
                .xlsx · .xls · .csv
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        new_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        st.markdown(f"<div class='bp-err'>❌ <strong>Could not read file:</strong> {e}</div>",
                    unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class='bp-info'>
        <strong>{uploaded.name}</strong> · {len(new_df):,} rows · {len(new_df.columns)} columns loaded.
    </div>
    """, unsafe_allow_html=True)

    # ── Schema validation ─────────────────────────────────────────────────────
    st.markdown('<div class="bp-section">Schema check</div>', unsafe_allow_html=True)
    schema = _validate_schema(new_df, X_train)

    n_missing = len(schema["missing"])
    n_extra   = len(schema["extra"])
    n_dtype   = len(schema["dtype_mismatch"])

    st.markdown(_stat_cards(
        ("Common cols",     str(len(schema["common"])),  "green"),
        ("Missing cols",    str(n_missing),              "amber" if n_missing else "green"),
        ("Extra cols",      str(n_extra),                "amber" if n_extra else "green"),
        ("Type mismatches", str(n_dtype),                "red"   if n_dtype else "green"),
    ), unsafe_allow_html=True)

    # Detailed schema report
    schema_html = ""
    for col in schema["common"][:5]:
        schema_html += _schema_row("✅", col, "match", "ok")
    if len(schema["common"]) > 5:
        schema_html += _schema_row("…", f"{len(schema['common']) - 5} more matching columns", "", "")
    for col in schema["missing"]:
        schema_html += _schema_row("⚠️", col, "missing — will be imputed", "warn")
    for col in schema["extra"]:
        schema_html += _schema_row("➖", col, "extra — will be ignored", "warn")
    for mis in schema["dtype_mismatch"]:
        schema_html += _schema_row("❌", mis["column"],
                                    f"got {mis['got']}, expected {mis['expected']}", "error")

    if schema_html:
        with st.expander("📋 Column-by-column schema check", expanded=(n_missing > 0 or n_dtype > 0)):
            st.markdown(schema_html, unsafe_allow_html=True)

    # Block if too many critical issues
    if n_dtype > 0:
        st.markdown(f"""
        <div class='bp-err'>
            ❌ <strong>{n_dtype} column(s) have incompatible data types.</strong>
            These need to be fixed before predictions can run. Most often this means a
            column expected to be numeric contains text values.
        </div>
        """, unsafe_allow_html=True)
        return

    if n_missing > X_train.shape[1] * 0.3:
        st.markdown(f"""
        <div class='bp-err'>
            ❌ <strong>{n_missing} of {X_train.shape[1]} columns are missing</strong> ({n_missing/X_train.shape[1]*100:.0f}%).
            Predictions made from imputed values would not be reliable. Please check your file.
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Prediction settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            fill_strategy = st.selectbox(
                "Imputation for missing cols",
                ["median", "mean"],
                key="bp_fill",
                help="How to fill missing columns using training-set statistics.",
            )
        with c2:
            batch_size = st.slider(
                "Batch size", 100, 10000, 1000, 100, key="bp_batch",
                help="Larger batches = faster but more memory. Lower if you hit memory limits.",
            )
        with c3:
            confidence_threshold = st.slider(
                "Low-confidence threshold", 0.0, 1.0, 0.6, 0.05, key="bp_conf",
                help="Predictions below this confidence will be flagged.",
            )
        include_original = st.checkbox(
            "Include original input columns in output", value=True, key="bp_incl",
        )

    # ── Run predictions ───────────────────────────────────────────────────────
    if st.button("🚀 Run predictions", use_container_width=True, type="primary", key="bp_run"):
        st.session_state.bp_run_clicked = True

    if not st.session_state.get("bp_run_clicked"):
        st.markdown("""
        <div style='text-align:center;padding:1.5rem;color:#e4e7ec;font-family:Space Mono,monospace;font-size:.8rem;'>
            Schema looks good — click <strong>Run predictions</strong> when ready ↑
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Align ─────────────────────────────────────────────────────────────────
    with st.spinner("🔧 Aligning data to model schema…"):
        X_aligned = _align_to_schema(new_df, X_train, fill_strategy=fill_strategy)

    # ── Predict in batches ────────────────────────────────────────────────────
    progress = st.progress(0.0, text="Predicting…")
    try:
        predictions, probabilities = _batched_predict(
            model, X_aligned, task, batch_size=batch_size,
            progress_cb=lambda p: progress.progress(p, text=f"Predicting… {int(p*100)}%"),
        )
        progress.empty()
    except Exception as e:
        progress.empty()
        st.markdown(f"<div class='bp-err'>❌ <strong>Prediction failed:</strong> {e}</div>",
                    unsafe_allow_html=True)
        return

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = _compute_confidence(predictions, probabilities, task, model=model, X=X_aligned)

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown('<div class="bp-section">Prediction summary</div>', unsafe_allow_html=True)

    n_low_conf = int((confidence < confidence_threshold).sum())
    pct_low    = n_low_conf / len(confidence) * 100

    if task == "classification":
        unique_preds = pd.Series(predictions).nunique()
        most_common  = pd.Series(predictions).mode().iloc[0]
        st.markdown(_stat_cards(
            ("Predictions",     f"{len(predictions):,}",        "blue"),
            ("Unique classes",  str(unique_preds),              "purple"),
            ("Most common",     str(most_common),               "green"),
            ("Mean confidence", f"{confidence.mean():.2f}",     "amber"),
            ("Low confidence",  f"{n_low_conf:,} ({pct_low:.1f}%)",
                                "red" if pct_low > 20 else "green"),
        ), unsafe_allow_html=True)
    else:
        st.markdown(_stat_cards(
            ("Predictions",   f"{len(predictions):,}",         "blue"),
            ("Mean",          f"{np.mean(predictions):.3g}",   "green"),
            ("Median",        f"{np.median(predictions):.3g}", "blue"),
            ("Min",           f"{np.min(predictions):.3g}",    "amber"),
            ("Max",           f"{np.max(predictions):.3g}",    "amber"),
            ("Low confidence", f"{n_low_conf:,} ({pct_low:.1f}%)",
                                "red" if pct_low > 20 else "green"),
        ), unsafe_allow_html=True)

    # ── Visualizations ────────────────────────────────────────────────────────
    fig = _prediction_dist(predictions, task)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    fig = _confidence_dist(confidence)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if pct_low > 20:
        st.markdown(f"""
        <div class='bp-warn'>
            ⚠️ <strong>{pct_low:.1f}% of predictions have low confidence.</strong>
            This often means your new data differs from what the model was trained on —
            consider retraining if this dataset is representative of current conditions.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='bp-good'>
            ✅ <strong>{100-pct_low:.1f}% of predictions are above the confidence threshold.</strong>
            Predictions look reliable for this dataset.
        </div>
        """, unsafe_allow_html=True)

    # ── Build output DataFrame ────────────────────────────────────────────────
    pred_col_name = f"{target_name}_predicted"
    if include_original:
        # Use the original (un-aligned) data so the user sees what they uploaded
        output_df = new_df.copy()
    else:
        output_df = pd.DataFrame(index=new_df.index)

    output_df[pred_col_name] = predictions
    output_df["confidence"]  = np.round(confidence, 4)
    output_df["flag"]        = np.where(confidence < confidence_threshold, "LOW_CONFIDENCE", "")

    # If classification with probabilities, add per-class probability columns
    if task == "classification" and probabilities is not None:
        try:
            est = model.named_steps[list(model.named_steps.keys())[-1]] if hasattr(model, "named_steps") else model
            classes = getattr(est, "classes_", np.arange(probabilities.shape[1]))
            for i, cls in enumerate(classes):
                output_df[f"proba_{cls}"] = np.round(probabilities[:, i], 4)
        except Exception:
            pass

    # ── Browse predictions ────────────────────────────────────────────────────
    st.markdown('<div class="bp-section">Browse predictions</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        filter_low = st.checkbox("Show only low-confidence rows", value=False, key="bp_filter_low")
    with c2:
        if task == "classification":
            filter_class = st.selectbox(
                "Filter by predicted class",
                ["All"] + sorted([str(c) for c in pd.Series(predictions).unique()]),
                key="bp_filter_class",
            )
        else:
            filter_class = "All"
            sort_pred = st.selectbox("Sort by prediction",
                                      ["No sort", "Highest first", "Lowest first"],
                                      key="bp_sort")
    with c3:
        max_show = st.slider("Rows to display", 10, 1000, 100, 10, key="bp_max_show")

    view_df = output_df.copy()
    if filter_low:
        view_df = view_df[view_df["confidence"] < confidence_threshold]
    if task == "classification" and filter_class != "All":
        view_df = view_df[view_df[pred_col_name].astype(str) == filter_class]
    if task == "regression":
        if sort_pred == "Highest first":
            view_df = view_df.sort_values(pred_col_name, ascending=False)
        elif sort_pred == "Lowest first":
            view_df = view_df.sort_values(pred_col_name, ascending=True)

    st.markdown(f"<div class='bp-info'>Showing <strong>{min(len(view_df), max_show):,}</strong> "
                f"of <strong>{len(view_df):,}</strong> rows.</div>",
                unsafe_allow_html=True)

    # Style: highlight low-confidence rows
    def _highlight_low(row):
        if row.get("flag") == "LOW_CONFIDENCE":
            return ["background-color: rgba(239,68,68,.08)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        view_df.head(max_show).style.apply(_highlight_low, axis=1)
               .format(precision=4, na_rep="—"),
        use_container_width=True,
        height=min(40 + min(len(view_df), max_show) * 35, 500),
    )

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("---")

    # Full Excel with predictions, low-confidence sheet, and summary
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Predictions")

        low_df = output_df[output_df["flag"] == "LOW_CONFIDENCE"]
        if not low_df.empty:
            low_df.to_excel(writer, index=False, sheet_name="Low confidence")

        # Summary sheet
        summary_rows = [
            ("Model",                model_name),
            ("Task",                 task),
            ("Predictions made",     len(predictions)),
            ("Mean confidence",      round(float(confidence.mean()), 4)),
            ("Min confidence",       round(float(confidence.min()),  4)),
            ("Low-confidence count", int(n_low_conf)),
            ("Low-confidence %",     round(pct_low, 2)),
            ("Imputed missing cols", n_missing),
            ("Dropped extra cols",   n_extra),
        ]
        if task == "regression":
            summary_rows += [
                ("Predicted mean",   round(float(np.mean(predictions)),   4)),
                ("Predicted median", round(float(np.median(predictions)), 4)),
                ("Predicted min",    round(float(np.min(predictions)),    4)),
                ("Predicted max",    round(float(np.max(predictions)),    4)),
            ]
        pd.DataFrame(summary_rows, columns=["Metric", "Value"]).to_excel(
            writer, index=False, sheet_name="Summary"
        )

    # CSV (sometimes easier to feed into other tools)
    csv_buf = output_df.to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download predictions (Excel, multi-sheet)",
            data=xlsx_buf.getvalue(),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="bp_dl_xlsx",
        )
    with d2:
        st.download_button(
            label="⬇️ Download predictions (CSV)",
            data=csv_buf,
            file_name=f"{uploaded.name.rsplit('.',1)[0]}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="bp_dl_csv",
        )