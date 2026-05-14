"""
ML AutoPilot — What-If Simulator widget
Drop-in module for the larger ML AutoPilot app.

Interactive prediction sandbox:
  • Edit any feature value via sliders / number inputs / dropdowns
  • See the prediction update instantly
  • Save scenarios and compare them side-by-side
  • Sensitivity analysis: sweep one feature across its range, plot the response
  • Plain-language summary of what changed and why

USAGE
---------------------------
    from what_if_simulator import render_what_if

    if page == "What-If Simulator":
        render_what_if(
            model=st.session_state.model,
            X_train=st.session_state.X_train,
            task=st.session_state.task,
            feature_names=st.session_state.feature_names,
            target_name=st.session_state.target,
        )
"""

import io
import json
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
    if st.session_state.get("_wi_css_done"):
        return
    st.markdown("""
    <style>
    .wi-section {
        color:#6b7280; font-family:'Space Mono',monospace;
        font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:2px;
        border-bottom:1px solid #f3f4f6; padding-bottom:.5rem; margin:1.8rem 0 1rem;
    }
    .wi-stat-row { display:flex; gap:.8rem; flex-wrap:wrap; margin:1rem 0; }
    .wi-stat-card {
        flex:1; min-width:130px;
        background:#ffffff; border:1px solid #e4e7ec; border-radius:8px;
        padding:1rem 1.2rem; text-align:center;
    }
    .wi-stat-card .s-label {
        color:#6b7280; font-size:.68rem; font-weight:600; text-transform:uppercase;
        letter-spacing:1.5px; font-family:'Space Mono',monospace; margin-bottom:.35rem;
    }
    .wi-stat-card .s-value { color:#1f2937; font-size:1.4rem; font-weight:700; font-family:'Space Mono',monospace; }
    .wi-stat-card .s-value.blue   { color:#4f46e5; }
    .wi-stat-card .s-value.green  { color:#10b981; }
    .wi-stat-card .s-value.amber  { color:#f59e0b; }
    .wi-stat-card .s-value.red    { color:#ef4444; }
    .wi-stat-card .s-value.purple { color:#8b5cf6; }

    /* HERO prediction card */
    .wi-hero {
        background: linear-gradient(120deg,#ffffff 0%,#111827 60%,#ffffff 100%);
        border: 1px solid #e4e7ec;
        border-left: 4px solid #8b5cf6;
        border-radius: 10px;
        padding: 1.6rem 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .wi-hero .hero-label {
        color:#6b7280; font-size:.72rem; font-weight:700;
        text-transform:uppercase; letter-spacing:2px;
        font-family:'Space Mono',monospace; margin-bottom:.4rem;
    }
    .wi-hero .hero-value {
        color:#1f2937; font-size:2.6rem; font-weight:800;
        font-family:'Space Mono',monospace; letter-spacing:-1px;
        line-height:1;
    }
    .wi-hero .hero-delta {
        display:inline-block; margin-left:1rem;
        font-family:'Space Mono',monospace; font-size:.95rem;
        font-weight:700; padding:.25rem .7rem; border-radius:6px;
        vertical-align: middle;
    }
    .wi-hero .hero-delta.up   { background:rgba(16,185,129,.12);  border:1px solid rgba(16,185,129,.3);  color:#10b981; }
    .wi-hero .hero-delta.down { background:rgba(239,68,68,.12);  border:1px solid rgba(239,68,68,.3);  color:#ef4444; }
    .wi-hero .hero-delta.same { background:rgba(107,114,128,.12);border:1px solid rgba(107,114,128,.3);color:#6b7280; }
    .wi-hero .hero-sub {
        color:#6b7280; font-size:.9rem; margin-top:.5rem;
    }

    /* boxes */
    .wi-info {
        background:rgba(79,70,229,.07); border:1px solid rgba(79,70,229,.2);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.6rem 0;
    }
    .wi-info strong { color:#4f46e5; }
    .wi-good {
        background:rgba(16,185,129,.07); border:1px solid rgba(16,185,129,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .wi-good strong { color:#10b981; }
    .wi-warn {
        background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .wi-warn strong { color:#f59e0b; }

    /* scenario chip */
    .wi-scenario {
        display:flex; align-items:center;
        background:#ffffff; border:1px solid #f3f4f6;
        border-radius:8px; padding:.6rem .9rem;
        margin-bottom:.4rem; gap:.8rem;
    }
    .wi-scenario .s-num {
        font-family:'Space Mono',monospace;
        font-size:.78rem; font-weight:700; color:#8b5cf6;
        width:24px; flex-shrink:0; text-align:center;
    }
    .wi-scenario .s-name { color:#1f2937; font-size:.92rem; flex:1; }
    .wi-scenario .s-val  {
        font-family:'Space Mono',monospace; font-size:.92rem;
        color:#4f46e5; font-weight:700;
    }

    /* change indicators in summary */
    .wi-change {
        background:#ffffff; border:1px solid #f3f4f6;
        border-radius:8px; padding:.6rem .9rem;
        margin-bottom:.35rem; font-size:.88rem;
    }
    .wi-change .c-feat { color:#1f2937; font-weight:600; }
    .wi-change .c-from { color:#6b7280; font-family:'Space Mono',monospace; }
    .wi-change .c-to   { color:#4f46e5; font-family:'Space Mono',monospace; font-weight:700; }
    .wi-change .c-arrow{ color:#e4e7ec; margin:0 .4rem; }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_wi_css_done"] = True


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
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    """Format a number for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if abs(v) < 0.01 and v != 0:
            return f"{v:.4g}"
        if abs(v) < 1000:
            return f"{v:.3f}".rstrip("0").rstrip(".")
        return f"{v:,.2f}"
    return str(v)


def _column_meta(X_train: pd.DataFrame, col: str) -> dict:
    """Compute slider bounds and type info for one column."""
    s = X_train[col].dropna()

    # Empty column — degenerate fallback
    if len(s) == 0:
        return {"type": "categorical", "options": [0], "default": 0}

    if pd.api.types.is_numeric_dtype(s):
        # Check if effectively categorical (few unique integers)
        n_unique = s.nunique()
        try:
            is_int = pd.api.types.is_integer_dtype(s) or bool((s % 1 == 0).all())
        except Exception:
            is_int = False

        if n_unique <= 1:
            # Constant column — show as locked single-option
            v = float(s.iloc[0])
            return {"type": "categorical_int", "options": [v], "default": v, "mean": v}

        if n_unique <= 10 and is_int:
            return {
                "type":    "categorical_int",
                "options": sorted(s.unique().tolist()),
                "default": int(s.median()),
                "mean":    float(s.mean()),
            }

        # Continuous numeric — slider
        p1, p99 = s.quantile([0.01, 0.99])
        p1, p99 = float(p1), float(p99)
        # Guard against degenerate ranges
        if not (p99 > p1):
            p99 = p1 + 1.0
        rng  = p99 - p1
        step = rng / 100 if rng > 0 else 0.01
        # Avoid step ≥ range (Streamlit rejects)
        if step >= rng:
            step = rng / 100

        return {
            "type":    "numeric",
            "min":     p1,
            "max":     p99,
            "default": float(np.clip(s.median(), p1, p99)),
            "mean":    float(s.mean()),
            "std":     float(s.std()),
            "step":    step,
            "is_int":  is_int,
        }

    # Object / categorical
    return {
        "type":    "categorical",
        "options": s.value_counts().head(50).index.tolist(),
        "default": s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
    }


def _predict_one(model, X_train: pd.DataFrame, row: dict, task: str):
    """Make a single prediction from a feature-value dict. Returns (value, proba_dict_or_None)."""
    df = pd.DataFrame([row], columns=X_train.columns)
    # Match dtypes to the training set so the pipeline doesn't complain
    for col in df.columns:
        try:
            df[col] = df[col].astype(X_train[col].dtype)
        except Exception:
            pass

    pred = model.predict(df)[0]
    proba = None
    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            probs   = model.predict_proba(df)[0]
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                last = list(model.named_steps.values())[-1]
                classes = getattr(last, "classes_", np.arange(len(probs)))
            proba = {str(c): float(p) for c, p in zip(classes, probs)}
        except Exception:
            pass
    return pred, proba


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _sensitivity_plot(model, X_train, base_row, feature, task, baseline_pred):
    """Sweep `feature` across its range, plot the predicted output."""
    _mpl_dark()
    meta = _column_meta(X_train, feature)

    if meta["type"] == "numeric":
        xs = np.linspace(meta["min"], meta["max"], 50)
    elif meta["type"] == "categorical_int":
        xs = np.array(meta["options"])
    else:
        return None   # can't sweep a string

    rows = []
    for x in xs:
        r = dict(base_row)
        r[feature] = float(x) if meta["type"] == "numeric" else x
        rows.append(r)

    df = pd.DataFrame(rows, columns=X_train.columns)
    for col in df.columns:
        try:
            df[col] = df[col].astype(X_train[col].dtype)
        except Exception:
            pass

    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df)
            # Use probability of the predicted class for the baseline
            y = probs[:, -1] if probs.shape[1] == 2 else probs.max(axis=1)
            y_label = "Predicted probability"
        except Exception:
            y = model.predict(df)
            y_label = "Predicted class"
    else:
        y = model.predict(df)
        y_label = "Predicted value"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(xs, y, color=BLUE, lw=2, alpha=0.85)
    ax.fill_between(xs, y, alpha=0.15, color=BLUE)

    # Mark the current value
    current_val = base_row[feature]
    ax.axvline(current_val, color=AMBER, lw=1.5, linestyle="--", alpha=0.8, label=f"Current = {_fmt(current_val)}")
    ax.axhline(baseline_pred, color=PURPLE, lw=1, linestyle=":", alpha=0.6, label=f"Current prediction")

    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"How '{feature}' affects the prediction", fontsize=11, fontweight="bold", color=TEXT)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _scenario_comparison_plot(scenarios: list, task: str):
    """Bar chart comparing scenario predictions."""
    if not scenarios:
        return None
    _mpl_dark()

    names = [s["name"] for s in scenarios]
    preds = [s["prediction"] for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.5)))
    fig.patch.set_facecolor(DARK_BG)

    colours = []
    base = preds[0]
    for p in preds:
        if abs(p - base) < 1e-9:
            colours.append(BLUE)
        elif p > base:
            colours.append(GREEN)
        else:
            colours.append(RED)

    ax.barh(names, preds, color=colours, alpha=0.85, edgecolor="none")
    ax.set_xlabel("Predicted value", fontsize=10)
    ax.set_title("Scenario comparison", fontsize=11, fontweight="bold", color=TEXT)
    ax.grid(True, alpha=0.2, axis="x")

    for i, v in enumerate(preds):
        ax.text(v, i, f"  {_fmt(v)}", va="center", fontsize=8, color=TEXT)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Stat-card / hero helpers
# ══════════════════════════════════════════════════════════════════════════════

def _stat_cards(*cards) -> str:
    html = "<div class='wi-stat-row'>"
    for label, value, colour in cards:
        html += f"""
        <div class='wi-stat-card'>
            <div class='s-label'>{label}</div>
            <div class='s-value {colour}'>{value}</div>
        </div>"""
    return html + "</div>"


def _hero_prediction(prediction, baseline, task, proba=None):
    """Render the big prediction card."""
    if baseline is None or task == "classification":
        delta_html = ""
    else:
        delta = prediction - baseline
        pct   = (delta / abs(baseline) * 100) if baseline != 0 else 0
        if abs(delta) < 1e-9:
            cls, sym = "same", "→"
        elif delta > 0:
            cls, sym = "up", "▲"
        else:
            cls, sym = "down", "▼"
        delta_html = f"<span class='hero-delta {cls}'>{sym} {_fmt(delta)} ({pct:+.1f}%)</span>"

    sub = ""
    if proba is not None:
        # Show top-2 classes with probabilities
        sorted_p = sorted(proba.items(), key=lambda x: -x[1])
        top_two  = sorted_p[:2]
        sub = "Probabilities: " + "  ·  ".join(f"<strong>{k}</strong>: {v*100:.1f}%" for k, v in top_two)

    return f"""
    <div class='wi-hero'>
        <div class='hero-label'>Current prediction</div>
        <div>
            <span class='hero-value'>{_fmt(prediction)}</span>
            {delta_html}
        </div>
        {f"<div class='hero-sub'>{sub}</div>" if sub else ""}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def _init_state(X_train: pd.DataFrame):
    if "wi_scenarios" not in st.session_state:
        st.session_state.wi_scenarios = []
    if "wi_current_row" not in st.session_state or st.session_state.get("wi_source_id") != id(X_train):
        # Initialize with median / mode for each column
        st.session_state.wi_current_row = {
            col: _column_meta(X_train, col)["default"] for col in X_train.columns
        }
        st.session_state.wi_baseline_row = dict(st.session_state.wi_current_row)
        st.session_state.wi_source_id    = id(X_train)


def _reset_to_baseline():
    st.session_state.wi_current_row = dict(st.session_state.wi_baseline_row)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_what_if(
    model=None,
    X_train: pd.DataFrame = None,
    task: str = "regression",
    feature_names: list = None,
    target_name: str = "target",
):
    """
    Render the What-If Simulator widget.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
    X_train : pandas.DataFrame — needed to compute slider bounds + dtypes
    task : "regression" | "classification"
    feature_names : list[str] — defaults to X_train.columns
    target_name : str — name of the predicted variable (display only)
    """
    _inject_css_once()

    st.markdown("## 🎯 What-If Simulator")
    st.caption("Edit any feature, see the prediction change instantly. Save and compare scenarios.")

    # ── Guard ─────────────────────────────────────────────────────────────────
    if model is None or X_train is None:
        st.markdown("""
        <div class='wi-warn'>
            ⚠️ <strong>No trained model in session.</strong>
            Run the AutoPilot module first, then come back here to play with predictions.
        </div>
        """, unsafe_allow_html=True)
        return

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    _init_state(X_train)

    # ── Baseline prediction (median row) ──────────────────────────────────────
    try:
        baseline_pred, _ = _predict_one(model, X_train, st.session_state.wi_baseline_row, task)
    except Exception as e:
        st.error(f"Could not generate baseline prediction: {e}")
        return

    # ── Importance ordering (so most-impactful features come first) ───────────
    # Use SHAP if available in session, else fall back to model.feature_importances_, else original order
    feat_order = X_train.columns.tolist()
    if "xp_importance" in st.session_state:
        try:
            feat_order = st.session_state.xp_importance.index.tolist() + \
                         [c for c in X_train.columns if c not in st.session_state.xp_importance.index]
        except Exception:
            pass
    else:
        try:
            est = model.named_steps[list(model.named_steps.keys())[-1]] if hasattr(model, "named_steps") else model
            if hasattr(est, "feature_importances_"):
                imp = pd.Series(est.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                feat_order = imp.index.tolist()
        except Exception:
            pass

    # ── Settings ──────────────────────────────────────────────────────────────
    n_feats = len(feat_order)
    with st.expander("⚙️ Simulator settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if n_feats <= 3:
                # Too few features — skip slider, show all
                top_k = n_feats
                st.caption(f"Only {n_feats} feature(s) — all shown.")
            else:
                slider_max     = min(30, n_feats)
                slider_default = min(10, slider_max)
                top_k = st.slider(
                    "Features to show as sliders",
                    min_value=3,
                    max_value=slider_max,
                    value=slider_default,
                    step=1,
                    key="wi_topk",
                    help="The rest will be locked at their median values. Order is by feature importance.",
                )
        with c2:
            show_locked = st.checkbox("Show locked features", value=False, key="wi_show_locked",
                                      help="Reveal the features kept at their median values.")

    shown_feats  = feat_order[:top_k]
    locked_feats = feat_order[top_k:]

    # ── Hero prediction card (computed against current row) ───────────────────
    try:
        current_pred, current_proba = _predict_one(
            model, X_train, st.session_state.wi_current_row, task
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    st.markdown(_hero_prediction(current_pred, baseline_pred, task, current_proba),
                unsafe_allow_html=True)

    # ── Input panel ───────────────────────────────────────────────────────────
    st.markdown('<div class="wi-section">Adjust inputs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='wi-info'>
        Slide or type to change a feature's value. The prediction above updates instantly.
        Features are ordered by <strong>importance</strong> — the ones that matter most appear first.
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout for inputs
    cols = st.columns(2)
    for i, feat in enumerate(shown_feats):
        with cols[i % 2]:
            meta = _column_meta(X_train, feat)
            current_val = st.session_state.wi_current_row.get(feat, meta["default"])

            if meta["type"] == "numeric":
                # Clamp current value into the slider's range to avoid Streamlit errors
                v = float(current_val) if current_val is not None else float(meta["default"])
                v = max(float(meta["min"]), min(float(meta["max"]), v))
                new_val = st.slider(
                    feat,
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=v,
                    step=float(meta["step"]),
                    key=f"wi_in_{feat}",
                    help=f"Mean: {_fmt(meta['mean'])} · Median: {_fmt(meta['default'])}",
                )
                if meta.get("is_int"):
                    new_val = int(round(new_val))
            elif meta["type"] == "categorical_int":
                new_val = st.select_slider(
                    feat,
                    options=meta["options"],
                    value=current_val if current_val in meta["options"] else meta["options"][0],
                    key=f"wi_in_{feat}",
                )
            else:
                new_val = st.selectbox(
                    feat,
                    options=meta["options"],
                    index=meta["options"].index(current_val) if current_val in meta["options"] else 0,
                    key=f"wi_in_{feat}",
                )

            st.session_state.wi_current_row[feat] = new_val

    if show_locked and locked_feats:
        with st.expander(f"🔒 Locked features ({len(locked_feats)})"):
            for feat in locked_feats:
                val = st.session_state.wi_current_row.get(feat, "—")
                st.text(f"{feat}: {_fmt(val)}")

    # ── Changes summary ───────────────────────────────────────────────────────
    changes = []
    for feat in shown_feats:
        base = st.session_state.wi_baseline_row.get(feat)
        curr = st.session_state.wi_current_row.get(feat)
        if base != curr:
            changes.append((feat, base, curr))

    if changes:
        st.markdown('<div class="wi-section">What you changed</div>', unsafe_allow_html=True)
        for feat, base, curr in changes:
            st.markdown(
                f"<div class='wi-change'>"
                f"<span class='c-feat'>{feat}</span>: "
                f"<span class='c-from'>{_fmt(base)}</span>"
                f"<span class='c-arrow'>→</span>"
                f"<span class='c-to'>{_fmt(curr)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Reset to baseline", use_container_width=True, key="wi_reset",
                     disabled=not changes):
            _reset_to_baseline()
            st.rerun()
    with c2:
        scenario_name = st.text_input("Scenario name", value=f"Scenario {len(st.session_state.wi_scenarios)+1}",
                                       key="wi_scn_name", label_visibility="collapsed",
                                       placeholder="Scenario name")
    with c3:
        if st.button("💾 Save scenario", use_container_width=True, type="primary", key="wi_save"):
            st.session_state.wi_scenarios.append({
                "name":       scenario_name or f"Scenario {len(st.session_state.wi_scenarios)+1}",
                "row":        dict(st.session_state.wi_current_row),
                "prediction": float(current_pred) if isinstance(current_pred, (int, float, np.integer, np.floating)) else str(current_pred),
                "proba":      current_proba,
            })
            st.toast(f"Saved '{scenario_name}'", icon="✅")
            st.rerun()

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    st.markdown('<div class="wi-section">Sensitivity analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='wi-info'>
        Pick a feature and see how the prediction would change if you swept it across its range —
        holding everything else at its current value. Look for <strong>plateaus</strong>
        ("after this point, changes don't matter") and <strong>cliffs</strong> ("here's where it flips").
    </div>
    """, unsafe_allow_html=True)

    sweep_feat = st.selectbox(
        "Feature to sweep",
        shown_feats,
        index=0,
        key="wi_sweep_feat",
    )

    meta = _column_meta(X_train, sweep_feat)
    if meta["type"] == "categorical":
        st.markdown(
            "<div class='wi-warn'>Sweeping is only available for numeric features. "
            "Pick a different feature.</div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            fig = _sensitivity_plot(
                model, X_train, st.session_state.wi_current_row,
                sweep_feat, task, current_pred,
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except Exception as e:
            st.markdown(f"<div class='wi-warn'>Sensitivity plot failed: {e}</div>",
                        unsafe_allow_html=True)

    # ── Saved scenarios ───────────────────────────────────────────────────────
    if st.session_state.wi_scenarios:
        st.markdown('<div class="wi-section">Saved scenarios</div>', unsafe_allow_html=True)

        # List with delete buttons
        for i, scn in enumerate(st.session_state.wi_scenarios):
            cs1, cs2, cs3, cs4 = st.columns([0.5, 3, 1.5, 1])
            with cs1:
                st.markdown(f"<div style='font-family:Space Mono,monospace;color:#8b5cf6;"
                            f"font-weight:700;padding-top:.6rem'>{i+1:02d}</div>",
                            unsafe_allow_html=True)
            with cs2:
                st.markdown(f"<div style='padding-top:.6rem;color:#1f2937'>{scn['name']}</div>",
                            unsafe_allow_html=True)
            with cs3:
                st.markdown(f"<div style='padding-top:.6rem;font-family:Space Mono,monospace;"
                            f"color:#4f46e5;font-weight:700;text-align:right'>{_fmt(scn['prediction'])}</div>",
                            unsafe_allow_html=True)
            with cs4:
                if st.button("✕", key=f"wi_del_{i}", help="Delete scenario"):
                    st.session_state.wi_scenarios.pop(i)
                    st.rerun()

        # Comparison plot
        if len(st.session_state.wi_scenarios) >= 1:
            # Prepend "current" for visual comparison
            comparison = [{"name": "Baseline (median)", "prediction": float(baseline_pred)
                                                            if isinstance(baseline_pred, (int, float, np.integer, np.floating))
                                                            else 0,
                           "row": st.session_state.wi_baseline_row}]
            comparison += st.session_state.wi_scenarios

            fig = _scenario_comparison_plot(comparison, task)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Detailed comparison table
        with st.expander("📋 Scenario details (all input values)"):
            rows = []
            for scn in st.session_state.wi_scenarios:
                r = dict(scn["row"])
                r["_scenario_"] = scn["name"]
                r["_prediction_"] = _fmt(scn["prediction"])
                rows.append(r)
            if rows:
                df_scn = pd.DataFrame(rows)
                cols_order = ["_scenario_", "_prediction_"] + [c for c in df_scn.columns
                                                                if c not in ("_scenario_", "_prediction_")]
                df_scn = df_scn[cols_order]
                st.dataframe(df_scn, use_container_width=True, hide_index=True)

        # Clear all
        if st.button("🗑️ Clear all scenarios", key="wi_clear", use_container_width=True):
            st.session_state.wi_scenarios = []
            st.rerun()

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("---")

    # Build scenario export
    if st.session_state.wi_scenarios:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            rows = []
            for scn in st.session_state.wi_scenarios:
                r = dict(scn["row"])
                r["scenario"]   = scn["name"]
                r["prediction"] = scn["prediction"]
                rows.append(r)
            pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Scenarios")

            # Also a wide-format comparison (one row per scenario)
            comparison_df = pd.DataFrame([
                {"scenario": s["name"], "prediction": s["prediction"], **s["row"]}
                for s in st.session_state.wi_scenarios
            ])
            comparison_df.to_excel(writer, index=False, sheet_name="Comparison")

        st.download_button(
            label="⬇️ Download scenarios (Excel)",
            data=buf.getvalue(),
            file_name=f"{target_name}_scenarios.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="wi_dl",
        )
    else:
        st.markdown(
            "<div class='wi-info'>💡 Save at least one scenario above to enable the download.</div>",
            unsafe_allow_html=True,
        )