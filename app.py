import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings

warnings.filterwarnings("ignore")

# basic page setup - wide layout feels much better for dashboards
st.set_page_config(
    page_title="Tidify · Clean & Visualize",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# injecting some custom styles because streamlit's defaults are a bit plain
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #161B27 0%, #1C2333 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-card h2 { color: #6EE7B7; font-size: 2rem; margin: 0; font-family: 'JetBrains Mono'; }
.metric-card p  { color: #8B949E; margin: 4px 0 0; font-size: 0.82rem;
                  text-transform: uppercase; letter-spacing: 0.08em; }

.clean-badge {
    display: inline-block; background: #0D2818;
    border: 1px solid #238636; border-radius: 6px;
    padding: 5px 12px; color: #3FB950;
    font-size: 0.8rem; margin: 4px 4px 4px 0;
    font-family: 'JetBrains Mono';
}
.action-badge {
    display: inline-block; background: #1C1700;
    border: 1px solid #9E6A03; border-radius: 6px;
    padding: 5px 12px; color: #D29922;
    font-size: 0.8rem; margin: 4px 4px 4px 0;
    font-family: 'JetBrains Mono';
}

.section-header {
    font-size: 1.1rem; font-weight: 700; color: #E6EDF3;
    border-left: 3px solid #6C63FF;
    padding-left: 12px; margin: 28px 0 14px;
}

.hero {
    text-align: center; padding: 90px 40px;
}
.hero .icon { font-size: 4rem; }
.hero h1 { color: #E6EDF3; font-size: 2.1rem; margin: 16px 0 8px; }
.hero p  { color: #8B949E; max-width: 480px; margin: auto;
           font-size: 1rem; line-height: 1.65; }
</style>
""", unsafe_allow_html=True)

# colors i picked manually - tried to keep them readable on dark backgrounds
PALETTE  = ["#6C63FF", "#43D4A0", "#FF6584", "#FFB347", "#4EC9FF",
            "#FF8C69", "#A78BFA", "#34D399", "#F472B6", "#FBBF24"]
BG  = "#0D1117"
AX  = "#161B27"

sns.set_theme(style="dark", rc={
    "axes.facecolor": AX, "figure.facecolor": BG,
    "text.color": "#C9D1D9", "axes.labelcolor": "#8B949E",
    "xtick.color": "#8B949E", "ytick.color": "#8B949E",
    "axes.edgecolor": "#30363D", "grid.color": "#21262D",
    "axes.titlecolor": "#E6EDF3",
})


# --- helper functions ---

def load_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f, engine="openpyxl")


def clean_dataframe(df: pd.DataFrame):
    log, actions = [], []
    df = df.copy()
    orig_r, orig_c = df.shape

    # people write "N/A", "null", empty string etc. - treat them all as NaN
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col].replace(
            {"nan": np.nan, "None": np.nan, "": np.nan,
             "N/A": np.nan, "n/a": np.nan, "NULL": np.nan, "null": np.nan,
             "NA": np.nan},
            inplace=True,
        )

    # rows/columns that are entirely empty are useless, just drop them
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    r_d = orig_r - df.shape[0]; c_d = orig_c - df.shape[1]
    if r_d: actions.append(f"Removed {r_d} empty rows")
    if c_d: actions.append(f"Removed {c_d} empty columns")

    # straightforward - if two rows are identical, keep one
    dupes = int(df.duplicated().sum())
    if dupes:
        df.drop_duplicates(inplace=True)
        actions.append(f"Dropped {dupes} duplicate rows")

    # sometimes numbers get imported as strings - try converting if 70%+ look numeric
    for col in list(df.select_dtypes(include="object").columns):
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().sum() / max(len(df), 1) > 0.7:
            df[col] = conv
            actions.append(f"Converted '{col}' → numeric")

    # guess date columns by their name - not perfect but catches the obvious ones
    for col in list(df.select_dtypes(include="object").columns):
        if any(k in col.lower() for k in ["date", "time", "year", "month", "day"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce",
                                         infer_datetime_format=True)
                actions.append(f"Parsed '{col}' → datetime")
            except Exception:
                pass

    # fill remaining gaps - median for numbers (robust to outliers), mode for text
    miss = int(df.isnull().sum().sum())
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        elif df[col].dtype == "object":
            mode = df[col].mode()
            if len(mode):
                df[col].fillna(mode[0], inplace=True)
    if miss:
        actions.append(f"Imputed {miss} missing values (median / mode)")

    if not actions:
        log.append("✅ Data was already clean!")
    return df, actions


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def make_charts(df: pd.DataFrame):
    numeric  = df.select_dtypes(include=np.number).columns.tolist()
    categoric = df.select_dtypes(include="object").columns.tolist()
    charts = []

    # ── 1. Missing values heatmap ─────────────────────────────────────────────
    total_miss = df.isnull().sum().sum()
    if total_miss > 0:
        miss_pct = df.isnull().mean() * 100
        miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(3, len(miss_pct) * 0.45)))
        ax.barh(miss_pct.index[::-1], miss_pct.values[::-1],
                color="#6C63FF", edgecolor=BG, height=0.65)
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values per Column (%)", fontsize=13, pad=12)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
        fig.tight_layout()
        charts.append(("🕳 Missing Values", fig))

    # ── 2. Distributions (histograms) ─────────────────────────────────────────
    if numeric:
        n = len(numeric); cols_n = min(3, n); rows_n = (n + 2) // 3
        fig, axes = plt.subplots(rows_n, cols_n,
                                 figsize=(14, rows_n * 3.6), facecolor=BG)
        axes = np.array(axes).flatten()
        for i, col in enumerate(numeric):
            ax = axes[i]; ax.set_facecolor(AX)
            data = df[col].dropna()
            ax.hist(data, bins=30, color=PALETTE[i % len(PALETTE)],
                    edgecolor=BG, alpha=0.88)
            mean_v = data.mean()
            ax.axvline(mean_v, color="#FF6584", lw=1.4, ls="--",
                       label=f"μ = {mean_v:.2f}")
            ax.legend(fontsize=8, facecolor=AX, labelcolor="#C9D1D9",
                      edgecolor="#30363D")
            ax.set_title(col, fontsize=10)
            for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
            ax.tick_params(labelsize=8)
        for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("Numeric Distributions", fontsize=14, color="#E6EDF3", y=1.01)
        fig.tight_layout()
        charts.append(("📊 Distributions", fig))

    # ── 3. Correlation heatmap ────────────────────────────────────────────────
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        sz = max(6, len(numeric))
        fig, ax = plt.subplots(figsize=(sz, sz * 0.78))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    mask=mask, ax=ax, linewidths=0.5, linecolor=BG,
                    annot_kws={"size": 8}, vmin=-1, vmax=1,
                    cbar_kws={"shrink": 0.75})
        ax.set_title("Correlation Heatmap", fontsize=13, pad=14)
        fig.tight_layout()
        charts.append(("🔥 Correlation", fig))

    # ── 4. Box plots ──────────────────────────────────────────────────────────
    if numeric:
        n = len(numeric); cols_n = min(3, n); rows_n = (n + 2) // 3
        fig, axes = plt.subplots(rows_n, cols_n,
                                 figsize=(14, rows_n * 3.6), facecolor=BG)
        axes = np.array(axes).flatten()
        for i, col in enumerate(numeric):
            ax = axes[i]; ax.set_facecolor(AX)
            data = df[col].dropna()
            ax.boxplot(data, patch_artist=True, widths=0.5,
                       medianprops=dict(color="#43D4A0", lw=2),
                       boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7),
                       whiskerprops=dict(color="#8B949E"),
                       capprops=dict(color="#8B949E"),
                       flierprops=dict(marker="o", color=PALETTE[i % len(PALETTE)],
                                       alpha=0.4, markersize=4))
            ax.set_title(col, fontsize=10)
            ax.set_xticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
            ax.tick_params(labelsize=8)
        for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("Box Plots — Spread & Outliers", fontsize=14,
                     color="#E6EDF3", y=1.01)
        fig.tight_layout()
        charts.append(("📦 Box Plots", fig))

    # ── 5. Bar charts for categorical columns ─────────────────────────────────
    bar_cols = [c for c in categoric if 2 <= df[c].nunique() <= 25][:4]
    for col in bar_cols:
        counts = df[col].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(counts.index.astype(str)[::-1], counts.values[::-1],
                color=PALETTE[:len(counts)], edgecolor=BG, height=0.65)
        ax.set_title(f"Value Counts — {col}", fontsize=13, pad=12)
        ax.set_xlabel("Count")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        charts.append((f"📊 {col}", fig))

    # ── 6. Scatter (first two numeric cols) ───────────────────────────────────
    if len(numeric) >= 2:
        x_c, y_c = numeric[0], numeric[1]
        fig, ax = plt.subplots(figsize=(9, 5))
        color_col = next((c for c in categoric if df[c].nunique() <= 12), None)
        if color_col:
            for idx, (nm, grp) in enumerate(df.groupby(color_col)):
                ax.scatter(grp[x_c], grp[y_c], color=PALETTE[idx % len(PALETTE)],
                           alpha=0.7, s=42, label=str(nm))
            ax.legend(title=color_col, fontsize=8, facecolor=AX,
                      labelcolor="#C9D1D9", edgecolor="#30363D", title_fontsize=8)
        else:
            ax.scatter(df[x_c], df[y_c], color=PALETTE[0], alpha=0.6, s=42)
        ax.set_xlabel(x_c); ax.set_ylabel(y_c)
        ax.set_title(f"Scatter — {x_c} vs {y_c}", fontsize=13, pad=12)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        charts.append(("✦ Scatter", fig))

    # ── 7. Pie chart ──────────────────────────────────────────────────────────
    pie_cols = [c for c in categoric if 2 <= df[c].nunique() <= 10]
    if pie_cols:
        col = pie_cols[0]
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
        ax.set_facecolor(BG)
        wedges, texts, autos = ax.pie(
            counts, labels=counts.index.astype(str),
            autopct="%1.1f%%", colors=PALETTE[:len(counts)],
            startangle=140, pctdistance=0.82,
            wedgeprops=dict(edgecolor=BG, linewidth=1.5),
        )
        for t in texts:  t.set_color("#C9D1D9"); t.set_fontsize(9)
        for a in autos:  a.set_fontsize(8); a.set_fontweight("bold"); a.set_color("#0D1117")
        ax.set_title(f"Share — {col}", fontsize=13, color="#E6EDF3", pad=14)
        fig.tight_layout()
        charts.append((f"🥧 {col}", fig))

    return charts


# --- sidebar: file upload + display toggles ---
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 20px'>
        <span style='font-size:2.4rem'>🧹</span>
        <h2 style='color:#E6EDF3;margin:6px 0 2px;font-size:1.4rem'>Tidify</h2>
        <p style='color:#8B949E;font-size:0.78rem;margin:0'>Clean · Explore · Visualize</p>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel up to 50 MB",
    )

    st.markdown("---")
    st.markdown("**View options**")
    show_raw     = st.checkbox("Raw data preview",     value=True)
    show_cleaned = st.checkbox("Cleaned data preview", value=True)
    show_stats   = st.checkbox("Descriptive stats",    value=True)
    run_charts   = st.checkbox("Generate charts",      value=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#8B949E;font-size:0.74rem;line-height:1.9'>
    <b style='color:#C9D1D9'>Cleaning steps</b><br>
    ✦ Normalise null strings<br>
    ✦ Drop empty rows & columns<br>
    ✦ Remove duplicates<br>
    ✦ Auto-convert numeric cols<br>
    ✦ Parse date columns<br>
    ✦ Impute missing values<br><br>
    <b style='color:#C9D1D9'>Charts generated</b><br>
    ✦ Missing-value map<br>
    ✦ Histograms<br>
    ✦ Correlation heatmap<br>
    ✦ Box plots<br>
    ✦ Bar charts (categorical)<br>
    ✦ Scatter plot<br>
    ✦ Pie chart
    </div>""", unsafe_allow_html=True)


# --- main content ---
if uploaded_file is None:
    st.markdown("""
    <div class='hero'>
        <div class='icon'>📂</div>
        <h1>Upload messy data. Get clean charts.</h1>
        <p>Drop a <b style='color:#6C63FF'>CSV</b> or
           <b style='color:#43D4A0'>Excel</b> file in the sidebar and Tidify
           will clean it up and visualize it automatically.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# load the file - handle both csv and excel
with st.spinner("Reading file …"):
    try:
        df_raw = load_file(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

# quick overview of what we're working with
st.markdown(f"## 📋 {uploaded_file.name}")
c1, c2, c3, c4 = st.columns(4)
for ui_col, val, label in [
    (c1, f"{df_raw.shape[0]:,}",                 "Total Rows"),
    (c2, f"{df_raw.shape[1]:,}",                 "Columns"),
    (c3, f"{df_raw.isnull().sum().sum():,}",      "Missing Cells"),
    (c4, f"{int(df_raw.duplicated().sum()):,}",   "Duplicate Rows"),
]:
    ui_col.markdown(f"""
    <div class='metric-card'>
        <h2>{val}</h2><p>{label}</p>
    </div>""", unsafe_allow_html=True)

# ── Raw preview
if show_raw:
    st.markdown("<div class='section-header'>Raw Data Preview</div>",
                unsafe_allow_html=True)
    st.dataframe(df_raw.head(100), use_container_width=True, height=280)

# run cleaning and show what changed
st.markdown("<div class='section-header'>🧹 Data Cleaning</div>",
            unsafe_allow_html=True)

with st.spinner("Cleaning …"):
    df_clean, actions = clean_dataframe(df_raw)

if actions:
    badges = "".join(f"<span class='action-badge'>{a}</span>" for a in actions)
else:
    badges = "<span class='clean-badge'>✅ Already clean</span>"
st.markdown(badges, unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Rows",   f"{df_clean.shape[0]:,}",
          delta=f"{df_clean.shape[0] - df_raw.shape[0]:+,}")
m2.metric("Columns", f"{df_clean.shape[1]:,}",
          delta=f"{df_clean.shape[1] - df_raw.shape[1]:+,}")
m3.metric("Missing cells remaining", f"{int(df_clean.isnull().sum().sum()):,}")

if show_cleaned:
    st.markdown("<div class='section-header'>Cleaned Data Preview</div>",
                unsafe_allow_html=True)
    st.dataframe(df_clean.head(100), use_container_width=True, height=280)

st.download_button(
    label="⬇️  Download Cleaned CSV",
    data=df_clean.to_csv(index=False).encode("utf-8"),
    file_name=f"cleaned_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
    mime="text/csv",
)

# basic stats - good sanity check before diving into charts
if show_stats:
    st.markdown("<div class='section-header'>📊 Descriptive Statistics</div>",
                unsafe_allow_html=True)
    num_desc = df_clean.describe(include=[np.number]).T.round(3)
    if not num_desc.empty:
        st.markdown("**Numeric columns**")
        st.dataframe(num_desc, use_container_width=True)
    cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        st.markdown("**Categorical columns**")
        st.dataframe(df_clean[cat_cols].describe().T, use_container_width=True)

# the fun part
if run_charts:
    st.markdown("<div class='section-header'>📈 Visualizations</div>",
                unsafe_allow_html=True)
    with st.spinner("Generating charts …"):
        charts = make_charts(df_clean)

    if not charts:
        st.info("No chartable columns found in this dataset.")
    else:
        tabs = st.tabs([name for name, _ in charts])
        for tab, (name, fig) in zip(tabs, charts):
            with tab:
                st.pyplot(fig, use_container_width=True)
                dl_name = name.replace(" ", "_").replace("/", "_") \
                              .encode("ascii", "ignore").decode() + ".png"
                st.download_button(
                    label=f"⬇️  Download chart",
                    data=fig_to_bytes(fig),
                    file_name=dl_name,
                    mime="image/png",
                    key=f"dl_{name}",
                )
                plt.close(fig)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#484F58;font-size:0.74rem'>"
    "Tidify · Streamlit + Matplotlib + Seaborn</p>",
    unsafe_allow_html=True,
)