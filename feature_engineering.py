"""
ML AutoPilot — Feature Engineering widget
Drop-in module for the larger ML AutoPilot app.

Guided no-code data transformations:
  • Missing value imputation (mean/median/mode/constant/drop)
  • Outlier handling (cap, remove, log-transform)
  • Scaling (standard, min-max, robust)
  • Encoding categoricals (label, one-hot, frequency)
  • Binning numeric columns
  • Creating interaction features (ratios, differences, products)
  • Date feature extraction (year/month/day/dayofweek/...)
  • Duplicate row removal

Every transform is recorded as a "recipe step" so the user can:
  - See what was applied, in order
  - Undo the last step
  - Reset to original
  - Download the recipe + transformed dataset

USAGE
---------------------------
    from feature_engineering import render_feature_engineering

    if page == "Feature Engineering":
        render_feature_engineering(df=st.session_state.df, filename=...)
"""

import io
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st

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
    if st.session_state.get("_fe_css_done"):
        return
    st.markdown("""
    <style>
    /* section header */
    .fe-section {
        color:#6b7280; font-family:'Space Mono',monospace;
        font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:2px;
        border-bottom:1px solid #f3f4f6; padding-bottom:.5rem; margin:1.8rem 0 1rem;
    }

    /* stat cards */
    .fe-stat-row { display:flex; gap:.8rem; flex-wrap:wrap; margin:1rem 0; }
    .fe-stat-card {
        flex:1; min-width:130px;
        background:#ffffff; border:1px solid #e4e7ec; border-radius:8px;
        padding:1rem 1.2rem; text-align:center;
    }
    .fe-stat-card .s-label {
        color:#6b7280; font-size:.68rem; font-weight:600; text-transform:uppercase;
        letter-spacing:1.5px; font-family:'Space Mono',monospace; margin-bottom:.35rem;
    }
    .fe-stat-card .s-value { color:#1f2937; font-size:1.4rem; font-weight:700; font-family:'Space Mono',monospace; }
    .fe-stat-card .s-value.blue   { color:#4f46e5; }
    .fe-stat-card .s-value.green  { color:#10b981; }
    .fe-stat-card .s-value.amber  { color:#f59e0b; }
    .fe-stat-card .s-value.purple { color:#8b5cf6; }

    /* boxes */
    .fe-info {
        background:rgba(79,70,229,.07); border:1px solid rgba(79,70,229,.2);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.6rem 0;
    }
    .fe-info strong { color:#4f46e5; }
    .fe-good {
        background:rgba(16,185,129,.07); border:1px solid rgba(16,185,129,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .fe-good strong { color:#10b981; }
    .fe-warn {
        background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .fe-warn strong { color:#f59e0b; }

    /* recipe step */
    .fe-recipe-row {
        display:flex; align-items:center;
        background:#ffffff; border:1px solid #f3f4f6;
        border-radius:8px; padding:.6rem .9rem;
        margin-bottom:.4rem; gap:.8rem;
    }
    .fe-recipe-num {
        font-family:'Space Mono',monospace;
        font-size:.78rem; font-weight:700;
        color:#8b5cf6; width:24px; flex-shrink:0; text-align:center;
    }
    .fe-recipe-op {
        font-family:'Space Mono',monospace;
        font-size:.75rem; color:#4f46e5;
        background:rgba(79,70,229,.1);
        padding:.15rem .55rem; border-radius:4px;
        text-transform:uppercase; letter-spacing:1px;
    }
    .fe-recipe-desc { color:#1f2937; font-size:.88rem; flex:1; }

    /* recommendation pill */
    .fe-rec {
        display:inline-block; background:rgba(139,92,246,.1);
        border:1px solid rgba(139,92,246,.3); color:#8b5cf6;
        border-radius:6px; padding:.3rem .7rem; font-size:.78rem;
        font-family:'Space Mono',monospace; margin:.2rem .2rem 0 0;
    }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_fe_css_done"] = True


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def _init_state(df: pd.DataFrame):
    """Set up session state for this widget's recipe pipeline."""
    if "fe_df_orig" not in st.session_state or st.session_state.get("fe_source_id") != id(df):
        # New dataset — initialize
        st.session_state.fe_df_orig    = df.copy()
        st.session_state.fe_df_current = df.copy()
        st.session_state.fe_recipe     = []        # list of step dicts
        st.session_state.fe_source_id  = id(df)


def _reset_pipeline():
    st.session_state.fe_df_current = st.session_state.fe_df_orig.copy()
    st.session_state.fe_recipe     = []


def _undo_last():
    if not st.session_state.fe_recipe:
        return
    # Replay all but last step
    df = st.session_state.fe_df_orig.copy()
    new_recipe = st.session_state.fe_recipe[:-1]
    for step in new_recipe:
        df = _apply_step(df, step)
    st.session_state.fe_df_current = df
    st.session_state.fe_recipe     = new_recipe


# ══════════════════════════════════════════════════════════════════════════════
# Transformation engine — pure functions
# ══════════════════════════════════════════════════════════════════════════════

def _apply_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Apply a single recipe step to the DataFrame."""
    op = step["op"]
    p  = step.get("params", {})

    if op == "impute":
        col, method = p["column"], p["method"]
        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
        elif method == "constant":
            df[col] = df[col].fillna(p["value"])
        elif method == "drop":
            df = df.dropna(subset=[col])

    elif op == "outlier_cap":
        col = p["column"]
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = s.clip(lower=lo, upper=hi)

    elif op == "outlier_remove":
        col = p["column"]
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df[(s >= lo) & (s <= hi)]

    elif op == "log_transform":
        col = p["column"]
        # Shift if there are non-positive values
        shift = max(0, -df[col].min() + 1) if df[col].min() <= 0 else 0
        df[col] = np.log(df[col] + shift)

    elif op == "scale":
        col, method = p["column"], p["method"]
        s = df[col]
        if method == "standard":
            df[col] = (s - s.mean()) / (s.std() + 1e-9)
        elif method == "minmax":
            df[col] = (s - s.min()) / (s.max() - s.min() + 1e-9)
        elif method == "robust":
            df[col] = (s - s.median()) / (s.quantile(0.75) - s.quantile(0.25) + 1e-9)

    elif op == "encode":
        col, method = p["column"], p["method"]
        if method == "label":
            df[col] = pd.Categorical(df[col].astype(str)).codes
        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        elif method == "frequency":
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)

    elif op == "bin":
        col, n_bins = p["column"], p["n_bins"]
        df[f"{col}_binned"] = pd.cut(df[col], bins=n_bins, labels=False, include_lowest=True)

    elif op == "interaction":
        a, b, kind = p["col_a"], p["col_b"], p["kind"]
        if kind == "product":
            df[f"{a}_x_{b}"] = df[a] * df[b]
        elif kind == "ratio":
            df[f"{a}_over_{b}"] = df[a] / (df[b].replace(0, np.nan))
        elif kind == "diff":
            df[f"{a}_minus_{b}"] = df[a] - df[b]
        elif kind == "sum":
            df[f"{a}_plus_{b}"] = df[a] + df[b]

    elif op == "date_features":
        col, parts = p["column"], p["parts"]
        dt = pd.to_datetime(df[col], errors="coerce")
        if "year"      in parts: df[f"{col}_year"]      = dt.dt.year
        if "month"     in parts: df[f"{col}_month"]     = dt.dt.month
        if "day"       in parts: df[f"{col}_day"]       = dt.dt.day
        if "dayofweek" in parts: df[f"{col}_dayofweek"] = dt.dt.dayofweek
        if "quarter"   in parts: df[f"{col}_quarter"]   = dt.dt.quarter
        if "is_weekend" in parts: df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    elif op == "drop_duplicates":
        df = df.drop_duplicates()

    elif op == "drop_column":
        df = df.drop(columns=[p["column"]], errors="ignore")

    elif op == "rename":
        df = df.rename(columns={p["old"]: p["new"]})

    return df


def _add_step(op: str, params: dict, description: str):
    """Append a step and apply it to the current df."""
    step = {"op": op, "params": params, "description": description}
    new_df = _apply_step(st.session_state.fe_df_current.copy(), step)
    st.session_state.fe_df_current = new_df
    st.session_state.fe_recipe.append(step)


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation engine
# ══════════════════════════════════════════════════════════════════════════════

def _recommendations(df: pd.DataFrame) -> list:
    """Generate smart starter recommendations."""
    recs = []

    # Missing values
    for col in df.columns:
        miss_pct = df[col].isnull().mean() * 100
        if 0 < miss_pct < 30:
            if pd.api.types.is_numeric_dtype(df[col]):
                recs.append({"op": "impute", "col": col,
                             "label": f"Fill {miss_pct:.0f}% missing in '{col}' with median"})
            else:
                recs.append({"op": "impute", "col": col,
                             "label": f"Fill {miss_pct:.0f}% missing in '{col}' with mode"})
        elif miss_pct >= 30:
            recs.append({"op": "drop_column", "col": col,
                         "label": f"Drop '{col}' ({miss_pct:.0f}% missing)"})

    # Categorical encoding
    for col in df.select_dtypes(include="object").columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 10:
            recs.append({"op": "encode", "col": col,
                         "label": f"One-hot encode '{col}' ({n_unique} categories)"})
        elif 10 < n_unique <= 50:
            recs.append({"op": "encode", "col": col,
                         "label": f"Label encode '{col}' ({n_unique} categories)"})

    # Duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        recs.append({"op": "drop_duplicates", "col": None,
                     "label": f"Drop {n_dupes} duplicate rows"})

    return recs[:6]   # cap at 6 to avoid clutter


# ══════════════════════════════════════════════════════════════════════════════
# Small UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _stat_cards(*cards) -> str:
    html = "<div class='fe-stat-row'>"
    for label, value, colour in cards:
        html += f"""
        <div class='fe-stat-card'>
            <div class='s-label'>{label}</div>
            <div class='s-value {colour}'>{value}</div>
        </div>"""
    return html + "</div>"


def _render_recipe():
    """Render the current pipeline as a numbered list."""
    if not st.session_state.fe_recipe:
        st.markdown(
            "<div class='fe-info'>No transformations applied yet. Pick one below to get started.</div>",
            unsafe_allow_html=True,
        )
        return

    html = ""
    for i, step in enumerate(st.session_state.fe_recipe, 1):
        op = step["op"].replace("_", " ")
        desc = step["description"]
        html += f"""
        <div class='fe-recipe-row'>
            <div class='fe-recipe-num'>{i:02d}</div>
            <div class='fe-recipe-op'>{op}</div>
            <div class='fe-recipe-desc'>{desc}</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Transformation forms — one per operation
# ══════════════════════════════════════════════════════════════════════════════

def _form_impute(df: pd.DataFrame):
    miss_cols = [c for c in df.columns if df[c].isnull().any()]
    if not miss_cols:
        st.markdown("<div class='fe-good'>✅ No missing values to handle.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        col = st.selectbox("Column", miss_cols, key="fe_imp_col",
                           format_func=lambda c: f"{c} ({df[c].isnull().mean()*100:.1f}% missing)")
    with c2:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        opts   = ["median", "mean", "mode", "constant", "drop"] if is_num else ["mode", "constant", "drop"]
        method = st.selectbox("Method", opts, key="fe_imp_method")
    with c3:
        st.write("")  # spacer
        st.write("")
        apply_btn = st.button("Apply", key="fe_imp_apply", use_container_width=True)

    const_value = None
    if method == "constant":
        const_value = st.text_input("Constant value (number or text)", value="0", key="fe_imp_val")
        if is_num:
            try:    const_value = float(const_value)
            except: const_value = 0.0

    if apply_btn:
        params = {"column": col, "method": method}
        if method == "constant":
            params["value"] = const_value
        desc = f"Imputed '{col}' with {method}" + (f" ({const_value})" if method == "constant" else "")
        _add_step("impute", params, desc)
        st.rerun()


def _form_outliers(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.markdown("<div class='fe-warn'>No numeric columns available.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        col = st.selectbox("Numeric column", num_cols, key="fe_out_col")
    with c2:
        method = st.selectbox(
            "Strategy", ["cap (IQR)", "remove rows", "log transform"],
            key="fe_out_method",
            help="Cap = clip to IQR fences · Remove = delete outlier rows · Log = log-transform values"
        )
    with c3:
        st.write(""); st.write("")
        apply_btn = st.button("Apply", key="fe_out_apply", use_container_width=True)

    # Preview impact
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < lo) | (s > hi)).sum()
    st.markdown(
        f"<div class='fe-info'>'{col}' has <strong>{n_out}</strong> outliers "
        f"({n_out/len(s)*100:.1f}%) outside [{lo:.3g}, {hi:.3g}]</div>",
        unsafe_allow_html=True,
    )

    if apply_btn:
        if method == "cap (IQR)":
            _add_step("outlier_cap", {"column": col}, f"Capped outliers in '{col}' to IQR fences")
        elif method == "remove rows":
            _add_step("outlier_remove", {"column": col}, f"Removed outlier rows from '{col}'")
        else:
            _add_step("log_transform", {"column": col}, f"Log-transformed '{col}'")
        st.rerun()


def _form_scale(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.markdown("<div class='fe-warn'>No numeric columns available.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        cols = st.multiselect("Columns to scale", num_cols, default=num_cols[:3], key="fe_scl_cols")
    with c2:
        method = st.selectbox(
            "Method", ["standard (z-score)", "minmax (0-1)", "robust (median/IQR)"],
            key="fe_scl_method",
            help="Standard: mean=0, std=1 · MinMax: [0,1] range · Robust: less sensitive to outliers"
        )
    with c3:
        st.write(""); st.write("")
        apply_btn = st.button("Apply", key="fe_scl_apply", use_container_width=True)

    if apply_btn and cols:
        m = method.split()[0]   # "standard" / "minmax" / "robust"
        for col in cols:
            _add_step("scale", {"column": col, "method": m}, f"Scaled '{col}' ({m})")
        st.rerun()


def _form_encode(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        st.markdown("<div class='fe-good'>✅ No categorical columns to encode.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        col = st.selectbox("Categorical column", cat_cols, key="fe_enc_col",
                           format_func=lambda c: f"{c} ({df[c].nunique()} unique)")
    with c2:
        method = st.selectbox(
            "Method", ["one-hot", "label", "frequency"],
            key="fe_enc_method",
            help="One-hot: new column per category · Label: integer codes · Frequency: replace with prevalence"
        )
    with c3:
        st.write(""); st.write("")
        apply_btn = st.button("Apply", key="fe_enc_apply", use_container_width=True)

    n_unique = df[col].nunique()
    if method == "one-hot" and n_unique > 20:
        st.markdown(
            f"<div class='fe-warn'>⚠️ One-hot encoding '{col}' will create {n_unique} new columns. "
            "Consider label or frequency encoding instead.</div>",
            unsafe_allow_html=True,
        )

    if apply_btn:
        m = "onehot" if method == "one-hot" else method
        _add_step("encode", {"column": col, "method": m}, f"{method.title()}-encoded '{col}'")
        st.rerun()


def _form_bin(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.markdown("<div class='fe-warn'>No numeric columns available.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        col = st.selectbox("Numeric column", num_cols, key="fe_bin_col")
    with c2:
        n_bins = st.slider("Number of bins", 2, 20, 5, key="fe_bin_n")
    with c3:
        st.write(""); st.write("")
        apply_btn = st.button("Apply", key="fe_bin_apply", use_container_width=True)

    if apply_btn:
        _add_step("bin", {"column": col, "n_bins": n_bins},
                  f"Binned '{col}' into {n_bins} equal-width bins → '{col}_binned'")
        st.rerun()


def _form_interaction(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.markdown("<div class='fe-warn'>Need at least 2 numeric columns.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        a = st.selectbox("Column A", num_cols, key="fe_int_a")
    with c2:
        b = st.selectbox("Column B", [c for c in num_cols if c != a], key="fe_int_b")
    with c3:
        kind = st.selectbox("Operation", ["product", "ratio", "diff", "sum"], key="fe_int_kind")
    with c4:
        st.write(""); st.write("")
        apply_btn = st.button("Apply", key="fe_int_apply", use_container_width=True)

    if apply_btn:
        op_sym = {"product": "×", "ratio": "÷", "diff": "−", "sum": "+"}[kind]
        _add_step("interaction", {"col_a": a, "col_b": b, "kind": kind},
                  f"Created '{a} {op_sym} {b}'")
        st.rerun()


def _form_dates(df: pd.DataFrame):
    date_cols = df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns.tolist()
    # Also look for object cols that parse as dates
    for col in df.select_dtypes(include="object").columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_cols.append(col)
        except Exception:
            pass

    if not date_cols:
        st.markdown("<div class='fe-warn'>No date columns detected.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([2, 3, 1])
    with c1:
        col = st.selectbox("Date column", date_cols, key="fe_date_col")
    with c2:
        parts = st.multiselect(
            "Features to extract",
            ["year", "month", "day", "dayofweek", "quarter", "is_weekend"],
            default=["year", "month", "dayofweek"],
            key="fe_date_parts",
        )
    with c3:
        st.write(""); st.write("")
        apply_btn = st.button("Apply", key="fe_date_apply", use_container_width=True)

    if apply_btn and parts:
        _add_step("date_features", {"column": col, "parts": parts},
                  f"Extracted {', '.join(parts)} from '{col}'")
        st.rerun()


def _form_drop_column(df: pd.DataFrame):
    c1, c2 = st.columns([4, 1])
    with c1:
        cols = st.multiselect("Columns to drop", df.columns.tolist(), key="fe_drop_cols")
    with c2:
        st.write(""); st.write("")
        apply_btn = st.button("Apply", key="fe_drop_apply", use_container_width=True)

    if apply_btn and cols:
        for c in cols:
            _add_step("drop_column", {"column": c}, f"Dropped column '{c}'")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_feature_engineering(df: pd.DataFrame = None, filename: str = None):
    """
    Render the Feature Engineering widget.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Pre-loaded DataFrame. If None, the widget shows its own file uploader.
    filename : str, optional
        Display / download name for the dataset.
    """
    _inject_css_once()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🔧 Feature Engineering")
    st.caption("Guided no-code transformations — every step is recorded so you can undo or reapply later.")

    # ── Get a DataFrame ───────────────────────────────────────────────────────
    if df is None:
        uploaded = st.file_uploader(
            "Upload your dataset (.xlsx, .xls, or .csv)",
            type=["xlsx", "xls", "csv"],
            key="fe_uploader",
        )
        if uploaded is None:
            st.markdown("""
            <div style='text-align:center;padding:3rem 2rem;color:#e4e7ec;
                        border:1px dashed #f3f4f6;border-radius:10px;margin-top:1rem;'>
                <div style='font-size:2.5rem;margin-bottom:1rem;'>📂</div>
                <div style='font-size:1rem;color:#6b7280;'>Drop a file to start engineering features</div>
            </div>
            """, unsafe_allow_html=True)
            return
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            filename = uploaded.name
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

    if filename is None:
        filename = "dataset"

    _init_state(df)
    work = st.session_state.fe_df_current

    # ── 1. Overview ───────────────────────────────────────────────────────────
    st.markdown('<div class="fe-section">Current state</div>', unsafe_allow_html=True)

    orig = st.session_state.fe_df_orig
    delta_rows = len(work) - len(orig)
    delta_cols = len(work.columns) - len(orig.columns)

    st.markdown(_stat_cards(
        ("Rows",         f"{len(work):,}",      "blue"),
        ("Δ Rows",       f"{delta_rows:+,}",    "green" if delta_rows >= 0 else "amber"),
        ("Columns",      str(len(work.columns)),"blue"),
        ("Δ Columns",    f"{delta_cols:+d}",    "green" if delta_cols >= 0 else "amber"),
        ("Missing",      f"{work.isnull().sum().sum():,}",
                         "green" if work.isnull().sum().sum() == 0 else "amber"),
        ("Steps applied",str(len(st.session_state.fe_recipe)),
                         "purple" if st.session_state.fe_recipe else "amber"),
    ), unsafe_allow_html=True)

    # ── 2. Quick recommendations ──────────────────────────────────────────────
    recs = _recommendations(work)
    if recs:
        st.markdown('<div class="fe-section">Smart recommendations</div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='fe-info'>One-click fixes based on your current data state. "
            "These are starting points — you can always undo.</div>",
            unsafe_allow_html=True,
        )
        rec_cols = st.columns(min(3, len(recs)))
        for i, rec in enumerate(recs):
            with rec_cols[i % 3]:
                if st.button(rec["label"], key=f"fe_rec_{i}", use_container_width=True):
                    if rec["op"] == "impute":
                        method = "median" if pd.api.types.is_numeric_dtype(work[rec["col"]]) else "mode"
                        _add_step("impute", {"column": rec["col"], "method": method},
                                  f"Imputed '{rec['col']}' with {method}")
                    elif rec["op"] == "drop_column":
                        _add_step("drop_column", {"column": rec["col"]},
                                  f"Dropped column '{rec['col']}'")
                    elif rec["op"] == "encode":
                        n_unique = work[rec["col"]].nunique()
                        method = "onehot" if n_unique <= 10 else "label"
                        _add_step("encode", {"column": rec["col"], "method": method},
                                  f"{method.title()}-encoded '{rec['col']}'")
                    elif rec["op"] == "drop_duplicates":
                        n = work.duplicated().sum()
                        _add_step("drop_duplicates", {}, f"Dropped {n} duplicate rows")
                    st.rerun()

    # ── 3. Transformation tabs ────────────────────────────────────────────────
    st.markdown('<div class="fe-section">Manual transformations</div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🩹 Missing values",
        "📊 Outliers",
        "📏 Scaling",
        "🔤 Encoding",
        "📦 Binning",
        "✖️ Interactions",
        "📅 Dates",
        "🗑️ Drop columns",
    ])
    with tabs[0]: _form_impute(work)
    with tabs[1]: _form_outliers(work)
    with tabs[2]: _form_scale(work)
    with tabs[3]: _form_encode(work)
    with tabs[4]: _form_bin(work)
    with tabs[5]: _form_interaction(work)
    with tabs[6]: _form_dates(work)
    with tabs[7]: _form_drop_column(work)

    # ── 4. Recipe ─────────────────────────────────────────────────────────────
    st.markdown('<div class="fe-section">Pipeline recipe</div>', unsafe_allow_html=True)
    _render_recipe()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↩️ Undo last", use_container_width=True,
                     disabled=not st.session_state.fe_recipe, key="fe_undo"):
            _undo_last()
            st.rerun()
    with c2:
        if st.button("🔄 Reset all", use_container_width=True,
                     disabled=not st.session_state.fe_recipe, key="fe_reset"):
            _reset_pipeline()
            st.rerun()
    with c3:
        # Promote to session_state so other widgets (AutoPilot, etc.) can pick up the transformed df
        if st.button("✅ Save to session", use_container_width=True,
                     type="primary", key="fe_save"):
            st.session_state.df = work.copy()
            st.session_state.filename = filename
            st.toast("Transformed dataset saved to session — other widgets will use it.", icon="✅")

    # ── 5. Preview transformed data ───────────────────────────────────────────
    with st.expander("📋 Preview transformed data (first 20 rows)"):
        st.dataframe(work.head(20), use_container_width=True)

    # ── 6. Downloads ──────────────────────────────────────────────────────────
    st.markdown("---")

    # Transformed Excel
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        work.to_excel(writer, index=False, sheet_name="Transformed")
        if st.session_state.fe_recipe:
            recipe_df = pd.DataFrame([
                {"step": i+1, "operation": s["op"], "description": s["description"],
                 "params": json.dumps(s["params"])}
                for i, s in enumerate(st.session_state.fe_recipe)
            ])
            recipe_df.to_excel(writer, index=False, sheet_name="Recipe")

    # Recipe JSON
    recipe_json = json.dumps(st.session_state.fe_recipe, indent=2, default=str)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download transformed data (Excel)",
            data=xlsx_buf.getvalue(),
            file_name=f"{filename.rsplit('.',1)[0]}_transformed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="fe_dl_xlsx",
        )
    with d2:
        st.download_button(
            label="⬇️ Download recipe (JSON)",
            data=recipe_json,
            file_name=f"{filename.rsplit('.',1)[0]}_recipe.json",
            mime="application/json",
            use_container_width=True,
            disabled=not st.session_state.fe_recipe,
            key="fe_dl_json",
            help="Reapply this exact pipeline to new data later."
        )