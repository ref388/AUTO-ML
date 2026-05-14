"""
ML AutoPilot — Data Explorer widget
Drop-in module for the larger ML AutoPilot app.

USAGE in your main app.py
---------------------------
    import streamlit as st
    from data_explorer import render_data_explorer

    page = st.sidebar.radio("Navigate", ["AutoPilot", "Data Explorer", ...])

    if page == "Data Explorer":
        render_data_explorer()

    # OR, if a DataFrame is already loaded elsewhere in your app:
    if page == "Data Explorer":
        render_data_explorer(df=st.session_state["df"], filename="my_data.xlsx")
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Design tokens (shared with the rest of ML AutoPilot) ──────────────────────
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
    """Inject Data Explorer styles. Safe to call multiple times — the flag in
    session_state prevents re-injection on reruns."""
    if st.session_state.get("_de_css_done"):
        return
    st.markdown("""
    <style>
    /* stat cards */
    .de-stat-row { display:flex; gap:.8rem; flex-wrap:wrap; margin:1rem 0; }
    .de-stat-card {
        flex:1; min-width:130px;
        background:#ffffff; border:1px solid #e4e7ec; border-radius:8px;
        padding:1rem 1.2rem; text-align:center;
    }
    .de-stat-card .s-label {
        color:#6b7280; font-size:.68rem; font-weight:600; text-transform:uppercase;
        letter-spacing:1.5px; font-family:'Space Mono',monospace; margin-bottom:.35rem;
    }
    .de-stat-card .s-value { color:#1f2937; font-size:1.4rem; font-weight:700; font-family:'Space Mono',monospace; }
    .de-stat-card .s-value.blue   { color:#4f46e5; }
    .de-stat-card .s-value.green  { color:#10b981; }
    .de-stat-card .s-value.amber  { color:#f59e0b; }
    .de-stat-card .s-value.red    { color:#ef4444; }
    .de-stat-card .s-value.purple { color:#8b5cf6; }

    /* info / warn / good boxes */
    .de-info {
        background:rgba(79,70,229,.07); border:1px solid rgba(79,70,229,.2);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.6rem 0;
    }
    .de-info strong { color:#4f46e5; }
    .de-warn {
        background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .de-warn strong { color:#f59e0b; }
    .de-good {
        background:rgba(16,185,129,.07); border:1px solid rgba(16,185,129,.25);
        border-radius:8px; padding:.9rem 1.2rem; color:#6b7280; font-size:.85rem; margin:.4rem 0;
    }
    .de-good strong { color:#10b981; }

    /* section header */
    .de-section {
        color:#6b7280; font-family:'Space Mono',monospace;
        font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:2px;
        border-bottom:1px solid #f3f4f6; padding-bottom:.5rem; margin:1.8rem 0 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_de_css_done"] = True


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
# Analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def _memory_str(df: pd.DataFrame) -> str:
    mem = df.memory_usage(deep=True).sum()
    if mem < 1024:    return f"{mem} B"
    if mem < 1024**2: return f"{mem/1024:.1f} KB"
    return f"{mem/1024**2:.2f} MB"


def _dtype_group(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
    return "categorical"


def _column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        grp  = _dtype_group(s)
        miss = s.isnull().sum()
        row  = {
            "Column":    col,
            "Type":      grp,
            "Missing":   miss,
            "Missing %": round(miss / len(s) * 100, 1),
            "Unique":    s.nunique(),
            "Unique %":  round(s.nunique() / len(s) * 100, 1),
        }
        if grp == "numeric":
            row.update({
                "Mean":   round(s.mean(),   4),
                "Std":    round(s.std(),    4),
                "Min":    round(s.min(),    4),
                "Median": round(s.median(), 4),
                "Max":    round(s.max(),    4),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((s < lo) | (s > hi)).sum()
        if n_out > 0:
            rows.append({
                "Column":    col,
                "Outliers":  int(n_out),
                "% of rows": round(n_out / len(s) * 100, 1),
                "IQR fence [low, high]": f"[{lo:.3g}, {hi:.3g}]",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Column", "Outliers", "% of rows", "IQR fence [low, high]"]
    )


def _suggestions(df: pd.DataFrame, profile: pd.DataFrame) -> list:
    tips = []

    for _, r in profile[profile["Missing %"] > 30].iterrows():
        tips.append({"level": "warn",
                     "text": f"'{r['Column']}' is {r['Missing %']}% empty — consider dropping or imputing before training."})

    low_var = profile[(profile["Unique"] <= 2) & (profile["Type"] == "numeric")]
    for _, r in low_var.iterrows():
        tips.append({"level": "warn",
                     "text": f"'{r['Column']}' has only {r['Unique']} unique values — it may not add predictive signal."})

    hi_card = profile[(profile["Type"] == "categorical") & (profile["Unique %"] > 50) & (profile["Unique"] > 20)]
    for _, r in hi_card.iterrows():
        tips.append({"level": "warn",
                     "text": f"'{r['Column']}' looks like a free-text or ID column ({r['Unique']} unique values). Encoding it directly may hurt the model."})

    dupes = df.duplicated().sum()
    if dupes > 0:
        tips.append({"level": "warn",
                     "text": f"{dupes} duplicate rows detected ({dupes/len(df)*100:.1f}%). Removing them is usually a good idea."})

    clean_cols = profile[(profile["Missing %"] == 0) & (profile["Unique"] > 1)]
    if len(clean_cols) == len(profile):
        tips.append({"level": "good", "text": "No missing values found — your dataset looks clean!"})
    elif len(clean_cols) >= len(profile) * 0.8:
        tips.append({"level": "good", "text": f"{len(clean_cols)} of {len(profile)} columns have zero missing values."})

    if len(df) < 200:
        tips.append({"level": "warn",
                     "text": f"Only {len(df)} rows — very small datasets make ML models unreliable. Try to collect more data."})
    elif len(df) >= 1000:
        tips.append({"level": "good", "text": f"{len(df):,} rows — solid dataset size for most ML tasks."})

    return tips


# ══════════════════════════════════════════════════════════════════════════════
# Plot builders
# ══════════════════════════════════════════════════════════════════════════════

def _missing_heatmap(df: pd.DataFrame):
    _mpl_dark()
    sample = df.isnull().astype(int)
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)

    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 0.6), 4))
    fig.patch.set_facecolor(DARK_BG)
    sns.heatmap(
        sample.T, ax=ax,
        cmap=sns.color_palette([PANEL_BG, RED], as_cmap=True),
        cbar=False, linewidths=0, xticklabels=False, yticklabels=True,
    )
    ax.set_xlabel("Rows (sample of up to 500)", fontsize=9)
    ax.set_ylabel("")
    ax.set_title("Missing value map — red = missing", fontsize=11, fontweight="bold", color=TEXT)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def _distribution_grid(df: pd.DataFrame, max_cols: int = 12):
    _mpl_dark()
    num_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    cat_cols = df.select_dtypes(include="object").columns.tolist()[:max_cols]
    all_cols = num_cols + cat_cols
    if not all_cols:
        return None

    ncols = min(4, len(all_cols))
    nrows = -(-len(all_cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.array(axes).flatten()

    for i, col in enumerate(all_cols):
        ax = axes[i]
        s = df[col].dropna()
        if col in num_cols:
            ax.hist(s, bins=30, color=BLUE, alpha=0.75, edgecolor="none")
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(s)
                xs  = np.linspace(s.min(), s.max(), 200)
                ax2 = ax.twinx()
                ax2.plot(xs, kde(xs), color=AMBER, lw=1.5, alpha=0.8)
                ax2.set_yticks([])
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
            except Exception:
                pass
            ax.set_xlabel(col, fontsize=8, color=SUBTEXT)
        else:
            top = s.astype(str).value_counts().head(10)
            ax.barh(top.index, top.values, color=PURPLE, alpha=0.75, edgecolor="none")
            ax.invert_yaxis()
            ax.set_xlabel("Count", fontsize=8, color=SUBTEXT)
            ax.tick_params(axis="y", labelsize=7)
        ax.set_title(col, fontsize=9, fontweight="bold", color=TEXT, pad=4)
        ax.set_ylabel("")
        ax.grid(True, alpha=0.2)

    for j in range(len(all_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Column distributions", fontsize=12, fontweight="bold", color=TEXT, y=1.01)
    fig.tight_layout()
    return fig


def _correlation_matrix(df: pd.DataFrame):
    _mpl_dark()
    num = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return None

    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.7), max(5, len(corr) * 0.6)))
    fig.patch.set_facecolor(DARK_BG)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap=cmap, vmin=-1, vmax=1, center=0,
        annot=len(corr) <= 15,
        fmt=".2f", annot_kws={"size": 8, "color": TEXT},
        linewidths=0.5, linecolor=BORDER,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation matrix (numeric columns)", fontsize=11, fontweight="bold", color=TEXT)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    return fig


def _outlier_strip(df: pd.DataFrame, max_cols: int = 8):
    _mpl_dark()
    num_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    if not num_cols:
        return None

    fig, axes = plt.subplots(1, len(num_cols), figsize=(max(10, len(num_cols) * 2.2), 4))
    fig.patch.set_facecolor(DARK_BG)
    if len(num_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, num_cols):
        s = df[col].dropna()
        ax.boxplot(
            s, vert=True, widths=0.5, positions=[0], patch_artist=True,
            boxprops=dict(color=BLUE, linewidth=1.2, facecolor=PANEL_BG),
            medianprops=dict(color=AMBER, linewidth=2),
            flierprops=dict(marker="o", markerfacecolor=RED, markersize=3, alpha=0.5, linestyle="none"),
            whiskerprops=dict(color=SUBTEXT, linewidth=1),
            capprops=dict(color=SUBTEXT, linewidth=1),
        )
        ax.set_xticks([0])
        ax.set_xticklabels([col], fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Outlier overview (IQR box plots)", fontsize=11, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Small HTML helpers
# ══════════════════════════════════════════════════════════════════════════════

def _stat_cards(*cards) -> str:
    html = "<div class='de-stat-row'>"
    for label, value, colour in cards:
        html += f"""
        <div class='de-stat-card'>
            <div class='s-label'>{label}</div>
            <div class='s-value {colour}'>{value}</div>
        </div>"""
    return html + "</div>"


def _tip_box(tip: dict) -> str:
    cls  = {"warn": "de-warn", "good": "de-good"}.get(tip["level"], "de-info")
    icon = {"warn": "⚠️",     "good": "✅"}.get(tip["level"],     "ℹ️")
    return f"<div class='{cls}'>{icon} {tip['text']}</div>"


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect date-like object columns."""
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
            except Exception:
                pass
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point — call this from your main app
# ══════════════════════════════════════════════════════════════════════════════

def render_data_explorer(df: pd.DataFrame = None, filename: str = None):
    """
    Render the Data Explorer widget.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Pre-loaded DataFrame. If None, the widget shows its own file uploader.
    filename : str, optional
        Display / download name for the dataset.
    """
    _inject_css_once()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🔍 Data Explorer")
    st.caption("Profile · Distributions · Correlations · Outliers — understand your data before you train")

    # ── Get a DataFrame ───────────────────────────────────────────────────────
    if df is None:
        uploaded = st.file_uploader(
            "Upload your dataset (.xlsx, .xls, or .csv)",
            type=["xlsx", "xls", "csv"],
            key="de_uploader",
            help="First row must be column headers."
        )
        if uploaded is None:
            st.markdown("""
            <div style='text-align:center;padding:3rem 2rem;color:#e4e7ec;
                        border:1px dashed #f3f4f6;border-radius:10px;margin-top:1rem;'>
                <div style='font-size:2.5rem;margin-bottom:1rem;'>📂</div>
                <div style='font-size:1rem;color:#6b7280;'>Drop a file to start exploring</div>
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

    df = df.copy()
    df = _try_parse_dates(df)

    # ── Settings (inline expander; doesn't fight AutoPilot's sidebar) ─────────
    with st.expander("⚙️ Explorer settings", expanded=False):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            col_filter = st.multiselect(
                "Columns to analyse",
                options=df.columns.tolist(),
                default=df.columns.tolist(),
                key="de_cols",
            )
        with c2:
            max_dist_cols = st.slider("Max dist. plots", 4, 24, 12, 4, key="de_dist")
        with c3:
            max_box_cols  = st.slider("Max box plots",   4, 16,  8, 4, key="de_box")
        show_raw = st.checkbox("Show raw data preview", value=False, key="de_raw")

    if col_filter:
        df = df[col_filter]

    if df.empty or len(df.columns) == 0:
        st.warning("No columns selected.")
        return

    profile = _column_profile(df)

    # ── 1. Overview ───────────────────────────────────────────────────────────
    st.markdown('<div class="de-section">Dataset overview</div>', unsafe_allow_html=True)

    n_num    = (profile["Type"] == "numeric").sum()
    n_cat    = (profile["Type"] == "categorical").sum()
    miss_pct = round(df.isnull().sum().sum() / df.size * 100, 1)

    st.markdown(_stat_cards(
        ("Rows",        f"{len(df):,}",       "blue"),
        ("Columns",     str(len(df.columns)), "blue"),
        ("Numeric",     str(n_num),           "green"),
        ("Categorical", str(n_cat),           "purple"),
        ("Missing",     f"{miss_pct}%",       "amber" if miss_pct > 5 else "green"),
        ("Memory",      _memory_str(df),      "amber"),
    ), unsafe_allow_html=True)

    if show_raw:
        with st.expander("📋 Raw data (first 100 rows)"):
            st.dataframe(df.head(100), use_container_width=True)

    # ── 2. Smart suggestions ──────────────────────────────────────────────────
    tips = _suggestions(df, profile)
    if tips:
        st.markdown('<div class="de-section">Smart suggestions</div>', unsafe_allow_html=True)
        for t in tips:
            st.markdown(_tip_box(t), unsafe_allow_html=True)

    # ── 3. Column profile table ───────────────────────────────────────────────
    st.markdown('<div class="de-section">Column profile</div>', unsafe_allow_html=True)
    cols = ["Column", "Type", "Missing", "Missing %", "Unique", "Unique %",
            "Mean", "Std", "Min", "Median", "Max"]
    display_df = profile[[c for c in cols if c in profile.columns]].copy()

    def _colour_missing(val):
        if isinstance(val, (int, float)):
            if val > 30: return "color: #ef4444"
            if val > 5:  return "color: #f59e0b"
        return ""

    # pandas 2.1+ renamed Styler.applymap → Styler.map; keep both for compatibility
    styler = display_df.style
    if hasattr(styler, "map"):
        styler = styler.map(_colour_missing, subset=["Missing %"])
    else:
        styler = styler.applymap(_colour_missing, subset=["Missing %"])
    st.dataframe(
        styler.format(precision=2, na_rep="—"),
        use_container_width=True,
        height=min(40 + len(display_df) * 35, 500),
    )

    # ── 4. Missing values map ─────────────────────────────────────────────────
    if df.isnull().any().any():
        st.markdown('<div class="de-section">Missing values map</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='de-info'>
            Each row of the map is one column of your dataset. <strong>Red cells</strong> are missing values.
        </div>""", unsafe_allow_html=True)
        fig = _missing_heatmap(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.markdown(
            "<div class='de-good'>✅ <strong>No missing values</strong> — every cell is filled.</div>",
            unsafe_allow_html=True,
        )

    # ── 5. Distributions ──────────────────────────────────────────────────────
    st.markdown('<div class="de-section">Distributions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='de-info'>
        Numeric columns: <strong>histogram</strong> (blue) + <strong>density curve</strong> (orange).
        Categorical columns: top-10 most common values.
    </div>""", unsafe_allow_html=True)
    fig = _distribution_grid(df, max_cols=max_dist_cols)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No plottable columns found.")

    # ── 6. Correlation matrix ─────────────────────────────────────────────────
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        st.markdown('<div class="de-section">Correlation matrix</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='de-info'>
            Values close to <strong>+1</strong> = strong positive correlation,
            close to <strong>−1</strong> = strong negative, near <strong>0</strong> = little linear relationship.
            Highly correlated features (|r| &gt; 0.9) are often redundant.
        </div>""", unsafe_allow_html=True)

        corr_abs   = num_df.corr().abs()
        high_pairs = [
            (c1, c2, round(corr_abs.loc[c1, c2], 3))
            for i, c1 in enumerate(corr_abs.columns)
            for c2 in corr_abs.columns[i+1:]
            if corr_abs.loc[c1, c2] > 0.9
        ]
        if high_pairs:
            pairs_str = ", ".join(f"<strong>{a}</strong>↔<strong>{b}</strong> ({v})" for a, b, v in high_pairs)
            st.markdown(
                f"<div class='de-warn'>⚠️ Highly correlated pairs (|r| > 0.9): {pairs_str}</div>",
                unsafe_allow_html=True,
            )

        fig = _correlation_matrix(df)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── 7. Outlier report ─────────────────────────────────────────────────────
    st.markdown('<div class="de-section">Outlier report</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='de-info'>
        <strong>IQR method</strong>: values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged.
        Red dots in the box plots are outliers.
    </div>""", unsafe_allow_html=True)

    out_df = _outlier_report(df)
    if not out_df.empty:
        st.dataframe(out_df, use_container_width=True, hide_index=True)
        fig = _outlier_strip(df, max_cols=max_box_cols)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.markdown(
            "<div class='de-good'>✅ No outliers detected using IQR fences.</div>",
            unsafe_allow_html=True,
        )

    # ── 8. Download profile ───────────────────────────────────────────────────
    st.markdown("---")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        profile.to_excel(writer, index=False, sheet_name="Column Profile")
        out_df.to_excel(writer, index=False, sheet_name="Outlier Report")
        df.describe(include="all").T.reset_index().to_excel(
            writer, index=False, sheet_name="Describe"
        )

    st.download_button(
        label="⬇️ Download full profile (Excel)",
        data=buf.getvalue(),
        file_name=f"{filename.rsplit('.',1)[0]}_profile.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="de_download",
    )