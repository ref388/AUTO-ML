"""
ML AutoPilot — Model Explainability widget
Drop-in module for the larger ML AutoPilot app.

Explain what's driving a model's predictions, in plain language:
  • Global feature importance (SHAP mean |value| bar chart)
  • SHAP summary plot (beeswarm — direction & magnitude per feature)
  • Per-prediction explanation (waterfall — top drivers for a single row)
  • Feature dependency plots (how a feature affects predictions)
  • Plain-language insights ("region is 3× more important than gross_margin")

Uses SHAP TreeExplainer for tree models, KernelExplainer fallback for others.

USAGE
---------------------------
    from model_explainability import render_explainability

    if page == "Explainability":
        render_explainability(
            model=st.session_state.model,           # trained sklearn Pipeline
            X_train=st.session_state.X_train,
            X_test=st.session_state.X_test,
            y_test=st.session_state.y_test,
            task=st.session_state.task,             # "regression" | "classification"
            feature_names=st.session_state.X.columns.tolist(),
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
    if st.session_state.get("_xp_css_done"):
        return
    st.markdown("""
    <style>
    .xp-section {
        color:#6b7280; font-family:'Space Mono',monospace;
        font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:2px;
        border-bottom:1px solid #f3f4f6; padding-bottom:.5rem; margin:1.8rem 0 1rem;
    }
    .xp-stat-row { display:flex; gap:.8rem; flex-wrap:wrap; margin:1rem 0; }
    .xp-stat-card {
        flex:1; min-width:130px;
        background:#ffffff; border:1px solid #e4e7ec; border-radius:8px;
        padding:1rem 1.2rem; text-align:center;
    }
    .xp-stat-card .s-label {
        color:#6b7280; font-size:.68rem; font-weight:600; text-transform:uppercase;
        letter-spacing:1.5px; font-family:'Space Mono',monospace; margin-bottom:.35rem;
    }
    .xp-stat-card .s-value { color:#1f2937; font-size:1.4rem; font-weight:700; font-family:'Space Mono',monospace; }
    .xp-stat-card .s-value.blue   { color:#4f46e5; }
    .xp-stat-card .s-value.green  { color:#10b981; }
    .xp-stat-card .s-value.amber  { color:#f59e0b; }
    .xp-stat-card .s-value.purple { color:#8b5cf6; }

    .xp-info {
        background:rgba(79,70,229,.07); border:1px solid rgba(79,70,229,.2);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.6rem 0;
    }
    .xp-info strong { color:#4f46e5; }
    .xp-good {
        background:rgba(16,185,129,.07); border:1px solid rgba(16,185,129,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .xp-good strong { color:#10b981; }
    .xp-warn {
        background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .xp-warn strong { color:#f59e0b; }

    /* insight cards */
    .xp-insight {
        background:#ffffff; border:1px solid #f3f4f6;
        border-left: 3px solid #8b5cf6;
        border-radius:8px; padding:.9rem 1.2rem;
        margin-bottom:.5rem;
    }
    .xp-insight strong { color:#8b5cf6; }
    .xp-insight .xp-i-text { color:#1f2937; font-size:.92rem; }

    /* feature row in importance list */
    .xp-feat-row {
        display:flex; align-items:center;
        background:#ffffff; border:1px solid #f3f4f6;
        border-radius:8px; padding:.6rem .9rem;
        margin-bottom:.4rem; gap:.8rem;
    }
    .xp-feat-rank {
        font-family:'Space Mono',monospace;
        font-size:.78rem; font-weight:700;
        color:#e4e7ec; width:24px; flex-shrink:0; text-align:center;
    }
    .xp-feat-rank.gold { color:#f59e0b; }
    .xp-feat-name { color:#1f2937; font-weight:600; font-size:.92rem; flex:1; }
    .xp-feat-val {
        font-family:'Space Mono',monospace;
        font-size:.85rem; color:#4f46e5; font-weight:700;
    }
    .xp-feat-bar-wrap { width:140px; background:#f3f4f6; border-radius:4px; height:6px; }
    .xp-feat-bar { height:6px; border-radius:4px; background:#4f46e5; }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_xp_css_done"] = True


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
# SHAP computation (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _compute_shap(model_id, _model, X_train, X_test, task, max_samples=200):
    """
    Compute SHAP values. Returns (shap_values, base_value, X_used, success_flag, message).

    We pass model_id (id(model)) as a hashable cache key — the underlying _model
    is excluded from hashing via the leading underscore.
    """
    try:
        import shap
    except ImportError:
        return None, None, None, False, "SHAP is not installed. Run: pip install shap"

    # Limit sample size to keep things responsive
    X_explain = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
    X_bg      = X_train.iloc[:100]       if len(X_train) > 100       else X_train

    # The model is a sklearn Pipeline (scaler + estimator). SHAP works on the
    # final estimator with already-transformed data.
    try:
        # Try to extract the final estimator from a Pipeline
        if hasattr(_model, "named_steps"):
            steps = list(_model.named_steps.values())
            estimator = steps[-1]
            # Transform X with all preceding steps
            X_explain_t = X_explain.copy()
            X_bg_t      = X_bg.copy()
            for step in steps[:-1]:
                X_explain_t = step.transform(X_explain_t)
                X_bg_t      = step.transform(X_bg_t)
        else:
            estimator   = _model
            X_explain_t = X_explain.values
            X_bg_t      = X_bg.values

        # Try TreeExplainer first (fast for tree models)
        tree_models = ("RandomForest", "GradientBoosting", "ExtraTrees", "XGB", "LGBM", "DecisionTree")
        is_tree = any(t in type(estimator).__name__ for t in tree_models)

        if is_tree:
            explainer   = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_explain_t)
            base_value  = explainer.expected_value
        else:
            # KernelExplainer fallback — slower but works for any model
            f = (estimator.predict_proba if task == "classification" and hasattr(estimator, "predict_proba")
                 else estimator.predict)
            explainer   = shap.KernelExplainer(f, X_bg_t)
            shap_values = explainer.shap_values(X_explain_t, nsamples=100, silent=True)
            base_value  = explainer.expected_value

        # Normalize multi-class output to 2D array
        if isinstance(shap_values, list):
            # Multi-class: list of [n_samples, n_features] arrays — take class 1 for binary, mean abs for multi
            if len(shap_values) == 2:
                shap_values = shap_values[1]
                base_value  = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
            else:
                # Multi-class: average across classes
                shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
                base_value  = np.mean(base_value) if isinstance(base_value, (list, np.ndarray)) else base_value
        elif shap_values.ndim == 3:
            # Newer SHAP returns (samples, features, classes) for multi-class
            if shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]
                base_value  = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
            else:
                shap_values = np.mean(np.abs(shap_values), axis=2)
                base_value  = np.mean(base_value) if isinstance(base_value, (list, np.ndarray)) else base_value

        return shap_values, base_value, X_explain, True, "ok"

    except Exception as e:
        return None, None, None, False, f"SHAP computation failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _importance_chart(importance: pd.Series, top_n: int = 15):
    """Horizontal bar chart of top features by mean |SHAP|."""
    _mpl_dark()
    top = importance.head(top_n).iloc[::-1]   # reverse for top-down display

    fig, ax = plt.subplots(figsize=(10, max(3, len(top) * 0.4)))
    fig.patch.set_facecolor(DARK_BG)

    # Color top 3 in amber, rest blue
    colours = [AMBER if i >= len(top) - 3 else BLUE for i in range(len(top))]
    ax.barh(top.index, top.values, color=colours, alpha=0.85, edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|  (impact on prediction)", fontsize=9)
    ax.set_title(f"Top {len(top)} feature importances", fontsize=11, fontweight="bold", color=TEXT)
    ax.grid(True, alpha=0.2, axis="x")

    # Annotate values
    for i, (name, val) in enumerate(top.items()):
        ax.text(val, i, f"  {val:.3f}", va="center", fontsize=8, color=SUBTEXT)

    fig.tight_layout()
    return fig


def _beeswarm_chart(shap_values: np.ndarray, X: pd.DataFrame, top_n: int = 12):
    """Beeswarm — each dot is a sample, x = SHAP impact, color = feature value."""
    _mpl_dark()
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    top_feats  = importance.head(top_n).index.tolist()[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_feats) * 0.45)))
    fig.patch.set_facecolor(DARK_BG)

    rng = np.random.default_rng(42)
    for i, feat in enumerate(top_feats):
        idx = X.columns.get_loc(feat)
        x   = shap_values[:, idx]
        vals= X[feat].values.astype(float)

        # Normalize feature values for color (cool=low, warm=high)
        v_min, v_max = np.nanpercentile(vals, [2, 98])
        if v_max > v_min:
            norm = np.clip((vals - v_min) / (v_max - v_min), 0, 1)
        else:
            norm = np.zeros_like(vals)

        # Custom colormap: blue → grey → red
        colours = np.array([
            np.interp(norm, [0, 0.5, 1], [88/255, 200/255, 248/255]),   # R
            np.interp(norm, [0, 0.5, 1], [166/255, 200/255, 81/255]),   # G
            np.interp(norm, [0, 0.5, 1], [255/255, 200/255, 73/255]),   # B
        ]).T

        # Jitter y for visibility
        y = i + rng.uniform(-0.3, 0.3, size=len(x))
        ax.scatter(x, y, c=colours, s=12, alpha=0.7, edgecolors="none")

    ax.axvline(0, color=SUBTEXT, lw=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=9)
    ax.set_title("Feature impact spread — color = feature value (blue=low, red=high)",
                 fontsize=11, fontweight="bold", color=TEXT)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return fig


def _waterfall_chart(shap_values: np.ndarray, X: pd.DataFrame, row_idx: int,
                     base_value: float, top_n: int = 10):
    """Waterfall — top drivers for a single prediction."""
    _mpl_dark()
    sv  = shap_values[row_idx]
    vals= X.iloc[row_idx]

    # Sort by absolute impact, keep top_n
    order = np.argsort(np.abs(sv))[::-1][:top_n]
    sv_top = sv[order]
    feat_names = X.columns[order]
    feat_vals  = vals.values[order]

    # Labels combine feature name + value
    labels = [f"{n} = {_fmt_num(v)}" for n, v in zip(feat_names, feat_vals)]

    # Reverse so largest is on top
    sv_top  = sv_top[::-1]
    labels  = labels[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    fig.patch.set_facecolor(DARK_BG)

    colours = [GREEN if v > 0 else RED for v in sv_top]
    ax.barh(labels, sv_top, color=colours, alpha=0.85, edgecolor="none")
    ax.axvline(0, color=SUBTEXT, lw=0.8)
    ax.set_xlabel("SHAP value  (← lowers prediction · raises prediction →)", fontsize=9)
    ax.set_title(f"Why row {row_idx} got its prediction", fontsize=11, fontweight="bold", color=TEXT)
    ax.grid(True, alpha=0.2, axis="x")

    for i, v in enumerate(sv_top):
        ax.text(v, i, f"  {v:+.3f}", va="center", fontsize=8,
                color=GREEN if v > 0 else RED)

    fig.tight_layout()
    return fig


def _dependency_chart(shap_values: np.ndarray, X: pd.DataFrame, feature: str):
    """Scatter: feature value (x) vs its SHAP value (y)."""
    _mpl_dark()
    idx = X.columns.get_loc(feature)
    x   = X[feature].values
    y   = shap_values[:, idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.scatter(x, y, s=18, alpha=0.6, color=BLUE, edgecolors="none")
    ax.axhline(0, color=SUBTEXT, lw=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel(f"{feature} (feature value)", fontsize=9)
    ax.set_ylabel(f"SHAP value for {feature}", fontsize=9)
    ax.set_title(f"How '{feature}' affects predictions", fontsize=11, fontweight="bold", color=TEXT)
    ax.grid(True, alpha=0.2)

    # Trend line (LOWESS-ish via rolling median over sorted values)
    try:
        sorted_idx = np.argsort(x)
        xs, ys = x[sorted_idx], y[sorted_idx]
        window = max(5, len(xs) // 20)
        smoothed = pd.Series(ys).rolling(window, center=True, min_periods=1).median().values
        ax.plot(xs, smoothed, color=AMBER, lw=2, alpha=0.8, label="Trend")
        ax.legend(fontsize=8)
    except Exception:
        pass

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plain-language insight generation
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_num(v):
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float) or isinstance(v, np.floating):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def _generate_insights(importance: pd.Series, shap_values: np.ndarray,
                       X: pd.DataFrame, task: str, target_name: str) -> list:
    """Generate plain-language insights from SHAP values."""
    tips = []

    # 1. Top driver
    top_feat = importance.index[0]
    top_imp  = importance.iloc[0]
    tips.append({
        "level": "primary",
        "text":  f"<strong>'{top_feat}'</strong> is the strongest driver of {target_name} "
                 f"in this model (mean impact = {top_imp:.3f})."
    })

    # 2. Top vs second
    if len(importance) > 1:
        second = importance.index[1]
        ratio  = importance.iloc[0] / (importance.iloc[1] + 1e-9)
        if ratio > 1.5:
            tips.append({
                "level": "primary",
                "text":  f"<strong>'{top_feat}'</strong> matters {ratio:.1f}× more than the "
                         f"next most important feature, <strong>'{second}'</strong>."
            })

    # 3. Concentration of importance
    total = importance.sum()
    top3_share = importance.head(3).sum() / (total + 1e-9) * 100
    if top3_share > 70:
        tips.append({
            "level": "primary",
            "text":  f"The top 3 features account for <strong>{top3_share:.0f}%</strong> "
                     f"of the model's decisions — predictions are concentrated."
        })
    elif top3_share < 35:
        tips.append({
            "level": "primary",
            "text":  f"Importance is spread across many features (top 3 = only {top3_share:.0f}%) "
                     f"— the model relies on a broad signal."
        })

    # 4. Direction of top feature
    idx     = X.columns.get_loc(top_feat)
    sv_top  = shap_values[:, idx]
    val_top = X[top_feat].values.astype(float)
    if np.std(val_top) > 0 and np.std(sv_top) > 0:
        corr = np.corrcoef(val_top, sv_top)[0, 1]
        if abs(corr) > 0.3:
            direction = "higher" if corr > 0 else "lower"
            effect    = f"raises {target_name}" if corr > 0 else f"lowers {target_name}"
            tips.append({
                "level": "primary",
                "text":  f"<strong>{direction.title()} '{top_feat}'</strong> generally {effect} "
                         f"(correlation between value and SHAP = {corr:+.2f})."
            })

    # 5. Low-importance features
    low_imp = importance[importance < importance.iloc[0] * 0.02]
    if len(low_imp) >= 3:
        tips.append({
            "level": "secondary",
            "text":  f"{len(low_imp)} features have negligible impact and could likely be dropped "
                     f"without hurting model performance."
        })

    return tips


# ══════════════════════════════════════════════════════════════════════════════
# Small UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _stat_cards(*cards) -> str:
    html = "<div class='xp-stat-row'>"
    for label, value, colour in cards:
        html += f"""
        <div class='xp-stat-card'>
            <div class='s-label'>{label}</div>
            <div class='s-value {colour}'>{value}</div>
        </div>"""
    return html + "</div>"


def _render_importance_list(importance: pd.Series, top_n: int = 10):
    """Render top features as a ranked list with bars."""
    top = importance.head(top_n)
    max_val = top.max()
    html = ""
    for rank, (name, val) in enumerate(top.items(), 1):
        pct = (val / (max_val + 1e-9)) * 100
        rank_cls = "gold" if rank == 1 else ""
        rank_sym = "🥇" if rank == 1 else f"{rank:02d}"
        html += f"""
        <div class='xp-feat-row'>
            <div class='xp-feat-rank {rank_cls}'>{rank_sym}</div>
            <div class='xp-feat-name'>{name}</div>
            <div class='xp-feat-bar-wrap'>
                <div class='xp-feat-bar' style='width:{pct:.0f}%'></div>
            </div>
            <div class='xp-feat-val'>{val:.4f}</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_explainability(
    model=None,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    y_test=None,
    task: str = "regression",
    feature_names: list = None,
    target_name: str = "target",
    max_samples: int = 200,
):
    """
    Render the Model Explainability widget.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
    X_train, X_test : pandas.DataFrame
    y_test : array-like (optional, used for prediction display)
    task : "regression" | "classification"
    feature_names : list[str] — column names of X (defaults to X_test.columns)
    target_name : str — name of the predicted variable (display only)
    max_samples : int — cap SHAP computation to this many test rows
    """
    _inject_css_once()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🔬 Model Explainability")
    st.caption("Understand what drives your predictions — global importance, per-row drivers, and plain-language insights.")

    # ── Guard: need a trained model ───────────────────────────────────────────
    if model is None or X_test is None or X_train is None:
        st.markdown("""
        <div class='xp-warn'>
            ⚠️ <strong>No trained model in session.</strong>
            Run the AutoPilot module first to train a model, then come back here to explain it.
        </div>
        """, unsafe_allow_html=True)
        return

    # Ensure DataFrames with feature names
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    if feature_names is None:
        feature_names = X_test.columns.tolist()

    # ── Sample size control ───────────────────────────────────────────────────
    with st.expander("⚙️ Explainability settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            max_samples = st.slider("SHAP sample size", 50, 500, max_samples, 50, key="xp_n",
                                    help="Larger samples = slower but more representative.")
        with c2:
            top_n = st.slider("Top features to display", 5, 25, 12, 1, key="xp_top")

    # ── Compute SHAP ──────────────────────────────────────────────────────────
    with st.spinner("🔬 Computing SHAP values… this can take a moment for non-tree models."):
        shap_values, base_value, X_used, ok, msg = _compute_shap(
            id(model), model, X_train, X_test, task, max_samples=max_samples
        )

    if not ok:
        st.markdown(f"""
        <div class='xp-warn'>
            ⚠️ <strong>Could not compute SHAP values:</strong> {msg}
        </div>
        """, unsafe_allow_html=True)
        return

    # Importance ranking
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_used.columns,
    ).sort_values(ascending=False)

    # Expose to other widgets (e.g. What-If Simulator orders features by importance)
    st.session_state.xp_importance = importance

    # ── 1. Overview ───────────────────────────────────────────────────────────
    st.markdown('<div class="xp-section">Model overview</div>', unsafe_allow_html=True)

    model_name = type(model.named_steps[list(model.named_steps.keys())[-1]]).__name__ \
                 if hasattr(model, "named_steps") else type(model).__name__

    st.markdown(_stat_cards(
        ("Model",         model_name,                "blue"),
        ("Task",          task.title(),              "purple"),
        ("Features",      str(X_used.shape[1]),      "blue"),
        ("Samples used",  f"{len(X_used):,}",        "amber"),
        ("Top driver",    importance.index[0],       "green"),
    ), unsafe_allow_html=True)

    # ── 2. Plain-language insights ────────────────────────────────────────────
    st.markdown('<div class="xp-section">Plain-language insights</div>', unsafe_allow_html=True)
    insights = _generate_insights(importance, shap_values, X_used, task, target_name)
    for ins in insights:
        st.markdown(
            f"<div class='xp-insight'><div class='xp-i-text'>{ins['text']}</div></div>",
            unsafe_allow_html=True,
        )

    # ── 3. Global importance ──────────────────────────────────────────────────
    st.markdown('<div class="xp-section">Feature importance ranking</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='xp-info'>
        This ranks features by their <strong>average impact</strong> on predictions
        (mean absolute SHAP value across all test samples). Higher = more influential.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        _render_importance_list(importance, top_n=min(top_n, 10))
    with col2:
        fig = _importance_chart(importance, top_n=top_n)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── 4. Beeswarm — direction & spread ──────────────────────────────────────
    st.markdown('<div class="xp-section">Impact spread (beeswarm)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='xp-info'>
        Each dot is one test sample. Position on the x-axis = its SHAP value
        (how much that feature pushed the prediction up or down).
        <strong>Color = feature value</strong> (blue = low, red = high).
        Wide spreads mean a feature has very different effects on different rows.
    </div>
    """, unsafe_allow_html=True)
    fig = _beeswarm_chart(shap_values, X_used, top_n=top_n)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── 5. Per-prediction explanation ─────────────────────────────────────────
    st.markdown('<div class="xp-section">Why this prediction? (per-row explainer)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='xp-info'>
        Pick any row to see <strong>which features pushed its prediction up</strong> (green)
        <strong>or down</strong> (red) compared to the average. This is the chart to show
        a customer or stakeholder when they ask "why did the model predict this?"
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        row_idx = st.number_input(
            "Row index", min_value=0, max_value=len(X_used) - 1, value=0, step=1, key="xp_row"
        )
    with c2:
        # Show actual + predicted side-by-side
        try:
            # Predict on the original X_test slice (before transform)
            pred = model.predict(X_used.iloc[[row_idx]])[0]
            pred_str = _fmt_num(pred)
        except Exception:
            pred_str = "—"
        actual_str = _fmt_num(y_test.iloc[row_idx] if hasattr(y_test, "iloc") else y_test[row_idx]) \
                     if y_test is not None else "—"
        sum_shap = shap_values[row_idx].sum()
        base_str = _fmt_num(float(base_value) if np.isscalar(base_value) else float(np.mean(base_value)))

        st.markdown(_stat_cards(
            ("Predicted", pred_str,   "blue"),
            ("Actual",    actual_str, "green"),
            ("Baseline",  base_str,   "amber"),
            ("Total SHAP", f"{sum_shap:+.3f}", "purple"),
        ), unsafe_allow_html=True)

    fig = _waterfall_chart(shap_values, X_used, row_idx,
                           float(base_value) if np.isscalar(base_value) else float(np.mean(base_value)),
                           top_n=10)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── 6. Dependency plot ────────────────────────────────────────────────────
    st.markdown('<div class="xp-section">Feature dependency</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='xp-info'>
        See how changing a single feature's value affects the model's prediction.
        Useful for spotting <strong>non-linear effects</strong> (e.g. "price only matters above $50").
    </div>
    """, unsafe_allow_html=True)

    selected_feat = st.selectbox(
        "Feature to inspect", importance.index.tolist(),
        index=0, key="xp_dep_feat"
    )
    fig = _dependency_chart(shap_values, X_used, selected_feat)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── 7. Downloads ──────────────────────────────────────────────────────────
    st.markdown("---")

    # Build a SHAP DataFrame for download
    shap_df = pd.DataFrame(shap_values, columns=X_used.columns, index=X_used.index)
    importance_df = importance.reset_index()
    importance_df.columns = ["Feature", "Mean |SHAP|"]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        importance_df.to_excel(writer, index=False, sheet_name="Importance")
        shap_df.head(500).to_excel(writer, sheet_name="SHAP values (first 500)")
        X_used.head(500).to_excel(writer, sheet_name="Feature values")

    st.download_button(
        label="⬇️ Download explainability report (Excel)",
        data=buf.getvalue(),
        file_name=f"{target_name}_explainability.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="xp_dl",
    )