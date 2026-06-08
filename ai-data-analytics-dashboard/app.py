import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title=" Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0d0f1a;
    color: #e2e8f0;
}

.main,
.block-container,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stMarkdownContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    background-color: transparent !important;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1220 0%, #131729 100%);
    border-right: 1px solid #1e2340;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #e2e8f0;
}

/* Sidebar radio bullets + bold options */
[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    gap: 0.7rem !important;
    background: transparent !important;
    color: #cbd5e1 !important;
    padding: 0.2rem 0.15rem !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: transparent !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #cbd5e1 !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    line-height: 1.35 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #ff4b4b !important;
    width: 1.05rem !important;
    height: 1.05rem !important;
}

.hero-header {
    background: linear-gradient(135deg, #1e2d5e 0%, #0f4c81 40%, #0d7377 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid #1e4080;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(13,115,119,0.3) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-header h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}

.hero-header p {
    color: #94cfd4;
    font-size: 0.95rem;
    margin: 0;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #38bdf8;
    border-left: 4px solid #0d7377;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #141827 0%, #1a2035 100%);
    border: 1px solid #1e2d50;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: #0d7377;
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}

.metric-card .label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

.insight-box {
    background: linear-gradient(135deg, #0f2027, #1a2d3a);
    border: 1px solid #1e3a4a;
    border-left: 4px solid #0d7377;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #b2d9df;
    font-size: 0.9rem;
}

.stSuccess > div {
    background: #042f1f !important;
    border-color: #059669 !important;
    color: #6ee7b7 !important;
}

.stWarning > div {
    background: #2d1f04 !important;
    border-color: #d97706 !important;
    color: #fcd34d !important;
}

.stInfo > div {
    background: #051e33 !important;
    border-color: #0284c7 !important;
    color: #7dd3fc !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1e2d50;
    border-radius: 8px;
    background-color: #141827 !important;
    color: #e2e8f0 !important;
}

[data-testid="stExpander"] {
    background-color: transparent !important;
    border: 1px solid #1e2d50 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] details {
    background-color: #0d0f1a !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    background-color: #141827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2d50 !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:active,
.streamlit-expanderHeader:hover,
.streamlit-expanderHeader:focus,
.streamlit-expanderHeader:active {
    background-color: #1a2035 !important;
    color: #e2e8f0 !important;
}

[data-testid="stExpander"] summary *,
[data-testid="stExpander"] details *,
.streamlit-expanderHeader,
.streamlit-expanderHeader * {
    color: #e2e8f0 !important;
    background-color: transparent !important;
}

.stDownloadButton > button,
.stButton > button {
    background: linear-gradient(135deg, #0f4c81, #0d7377) !important;
    color: #e2e8f0 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: opacity 0.2s !important;
}

.stDownloadButton > button:hover,
.stButton > button:hover,
.stDownloadButton > button:focus,
.stButton > button:focus,
.stDownloadButton > button:active,
.stButton > button:active {
    opacity: 0.85 !important;
    color: #e2e8f0 !important;
    background: linear-gradient(135deg, #0f4c81, #0d7377) !important;
}

.stSelectbox > div > div,
.stSlider > div,
.stTextInput > div > div,
.stNumberInput > div > div,
.stTextArea > div > div,
[data-baseweb="select"],
[data-baseweb="input"],
[data-baseweb="textarea"] {
    background-color: #141827 !important;
    border-color: #1e2d50 !important;
    color: #e2e8f0 !important;
}

input,
textarea,
select {
    background-color: #141827 !important;
    color: #e2e8f0 !important;
    border-color: #1e2d50 !important;
}

div[role="listbox"],
ul[role="listbox"],
[data-baseweb="popover"] {
    background-color: #141827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2d50 !important;
}

div[role="option"] {
    background-color: #141827 !important;
    color: #e2e8f0 !important;
}

div[role="option"]:hover,
div[role="option"]:focus,
div[role="option"]:active {
    background-color: #1a2035 !important;
    color: #e2e8f0 !important;
}

[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="textarea"] * {
    color: #e2e8f0 !important;
}

[data-testid="stFileUploader"] {
    background-color: transparent !important;
    color: #e2e8f0 !important;
}

[data-testid="stFileUploader"] section {
    background-color: #141827 !important;
    border: 1px dashed #1e2d50 !important;
    color: #e2e8f0 !important;
}

[data-testid="stFileUploader"] div {
    background-color: transparent !important;
    color: #e2e8f0 !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] button:active {
    background: linear-gradient(135deg, #0f4c81, #0d7377) !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2d50 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] * {
    color: #e2e8f0 !important;
}

button,
button:hover,
button:focus,
button:active,
div[role="button"],
div[role="button"]:hover,
div[role="button"]:focus,
div[role="button"]:active {
    color: #e2e8f0 !important;
}

code {
    background-color: #1a2035 !important;
    color: #38bdf8 !important;
}

.footer {
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
    padding: 2rem 0 0.5rem;
    border-top: 1px solid #1e2340;
    margin-top: 3rem;
}

.element-container .stPlotlyChart, 
.element-container .stImage {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":  "#141827",
    "axes.facecolor":    "#141827",
    "axes.edgecolor":    "#1e2d50",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e2d50",
    "grid.alpha":        0.5,
    "axes.titlecolor":   "#e2e8f0",
    "figure.edgecolor":  "#141827",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
PALETTE = ["#38bdf8", "#0d7377", "#7c3aed", "#f59e0b", "#10b981", "#f43f5e"]

def get_df():
    return st.session_state.get("cleaned_df", None)

def get_raw():
    return st.session_state.get("raw_df", None)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>📊</div>
        <div style='font-size:1.1rem; font-weight:700; color:#38bdf8;'>AI Data Dashboard</div>
    </div>
    <hr style='border-color:#1e2d50; margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠 Home",
        "📋 Dataset Understanding",
        "🧹 Data Cleaning",
        "🔍 EDA",
        "📈 Visualizations",
        "🤖 Prediction",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2d50; margin: 1rem 0;'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            st.session_state["raw_df"] = raw.copy()
            if "cleaned_df" not in st.session_state:
                st.session_state["cleaned_df"] = raw.copy()
            st.success(f"✓ Loaded — {raw.shape[0]} rows × {raw.shape[1]} cols")
        except Exception as e:
            st.error(f"Error: {e}")

def hero(title, subtitle):
    st.markdown(f"""
    <div class='hero-header'>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

def metric_card(col, value, label):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='value'>{value}</div>
        <div class='label'>{label}</div>
    </div>""", unsafe_allow_html=True)

def no_data_msg():
    st.info("📂 Upload a CSV dataset from the sidebar to get started.")

if page == "🏠 Home":
    hero("AI Data Analysis & Prediction Dashboard",
         "Upload any CSV dataset and explore, clean, visualize, and predict — all in one place.")

    c1, c2, c3, c4 = st.columns(4)
    df = get_df()
    metric_card(c1, df.shape[0] if df is not None else "—", "Records")
    metric_card(c2, df.shape[1] if df is not None else "—", "Features")
    metric_card(c3, df.isnull().sum().sum() if df is not None else "—", "Missing Values")
    metric_card(c4, df.duplicated().sum() if df is not None else "—", "Duplicates")

    st.markdown("<br>", unsafe_allow_html=True)
    section("How to Use This Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='insight-box'>
        <b>Step 1 — Upload</b><br>
        Use the sidebar to upload any CSV file. The dataset will be loaded automatically.
        </div>
        <div class='insight-box'>
        <b>Step 2 — Understand</b><br>
        Go to <i>Dataset Understanding</i> to see structure, dtypes, and missing values.
        </div>
        <div class='insight-box'>
        <b>Step 3 — Clean</b><br>
        Navigate to <i>Data Cleaning</i> to handle nulls, duplicates, and encode categories.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='insight-box'>
        <b>Step 4 — Explore</b><br>
        Open <i>EDA</i> for stats, correlations, and outlier detection.
        </div>
        <div class='insight-box'>
        <b>Step 5 — Visualize</b><br>
        Use <i>Visualizations</i> to generate charts: histogram, scatter, boxplot, and more.
        </div>
        <div class='insight-box'>
        <b>Step 6 — Predict</b><br>
        Head to <i>Prediction</i> to train a Linear Regression model and evaluate performance.
        </div>
        """, unsafe_allow_html=True)

elif page == "📋 Dataset Understanding":
    hero("Dataset Understanding", "Explore the structure and contents of your dataset.")
    df = get_raw()
    if df is None:
        no_data_msg()
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        miss_total = df.isnull().sum().sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        metric_card(c1, df.shape[0], "Rows")
        metric_card(c2, df.shape[1], "Columns")
        metric_card(c3, len(num_cols), "Numeric Cols")
        metric_card(c4, len(cat_cols), "Categorical Cols")
        metric_card(c5, miss_total, "Missing Vals")

        st.markdown("<br>", unsafe_allow_html=True)

        st.info(f"📌 This dataset has **{df.shape[0]} records** and **{df.shape[1]} features**. "
                f"It contains **{len(num_cols)} numeric** and **{len(cat_cols)} categorical** columns. "
                f"Total missing values: **{miss_total}**.")

        section("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            section("Column Names & Data Types")
            dtype_df = pd.DataFrame({"Column": df.columns, "DType": df.dtypes.values})
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        with col2:
            section("Missing Values per Column")
            miss = df.isnull().sum().reset_index()
            miss.columns = ["Column", "Missing"]
            miss["Pct"] = (miss["Missing"] / len(df) * 100).round(2)
            st.dataframe(miss[miss["Missing"] > 0] if miss["Missing"].sum() > 0 else miss,
                         use_container_width=True, hide_index=True)

        with st.expander("🔢 Numeric Columns"):
            st.write(num_cols if num_cols else "None found")

        with st.expander("🔤 Categorical Columns"):
            st.write(cat_cols if cat_cols else "None found")

        with st.expander("📊 Full Statistical Summary"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

elif page == "🧹 Data Cleaning":
    hero("Data Cleaning & Preprocessing", "Handle missing values, duplicates, and encode categories.")
    raw = get_raw()
    if raw is None:
        no_data_msg()
    else:
        df = raw.copy()

        miss_before = df.isnull().sum().sum()
        dups_before = df.duplicated().sum()

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

        miss_after = df.isnull().sum().sum()

        df.drop_duplicates(inplace=True)
        dups_removed = dups_before - df.duplicated().sum()

        le = LabelEncoder()
        enc_cols = []
        for col in df.select_dtypes(exclude=np.number).columns:
            df[col] = le.fit_transform(df[col].astype(str))
            enc_cols.append(col)

        st.session_state["cleaned_df"] = df.copy()

        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, miss_before, "Missing Before")
        metric_card(c2, miss_after, "Missing After")
        metric_card(c3, int(dups_removed), "Duplicates Removed")
        metric_card(c4, len(enc_cols), "Cols Encoded")

        st.markdown("<br>", unsafe_allow_html=True)
        st.success("✅ Dataset cleaned and saved. All pages will now use the cleaned version.")

        col1, col2 = st.columns(2)
        with col1:
            section("Missing Values Summary")
            miss_df = pd.DataFrame({
                "Stage": ["Before Cleaning", "After Cleaning"],
                "Missing Values": [miss_before, miss_after]
            })
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

        with col2:
            section("Encoded Columns")
            if enc_cols:
                st.dataframe(pd.DataFrame({"Encoded Column": enc_cols}),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No categorical columns found.")

        section("Cleaned Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Download Options")
        c1, c2 = st.columns(2)
        with c1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Cleaned Dataset", csv_data,
                               "cleaned_dataset.csv", "text/csv")
        with c2:
            stats_data = df.describe(include="all").T.to_csv().encode("utf-8")
            st.download_button("⬇️ Download Statistics", stats_data,
                               "dataset_statistics.csv", "text/csv")

elif page == "🔍 EDA":
    hero("Exploratory Data Analysis", "Uncover patterns, correlations, and outliers in your data.")
    df = get_df()
    if df is None:
        no_data_msg()
    else:
        num_df = df.select_dtypes(include=np.number)
        if num_df.empty:
            st.warning("No numeric columns found for EDA.")
        else:
            section("Descriptive Statistics")
            stats = num_df.agg(["mean", "median", "std", "min", "max"]).T
            stats.columns = ["Mean", "Median", "Std Dev", "Min", "Max"]
            stats["Mode"] = num_df.mode().iloc[0]
            st.dataframe(stats.round(3), use_container_width=True)

            section("Correlation Analysis")
            corr = num_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                        cmap="coolwarm", linewidths=0.5,
                        annot_kws={"size": 8},
                        cbar_kws={"shrink": 0.8})
            ax.set_title("Correlation Heatmap", pad=12, fontsize=13)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            section("Outlier Detection — Boxplots")
            cols = num_df.columns.tolist()
            n_per_row = 3
            rows = [cols[i:i+n_per_row] for i in range(0, len(cols), n_per_row)]
            for row_cols in rows:
                r_cols = st.columns(len(row_cols))
                for idx, col in enumerate(row_cols):
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.boxplot(df[col].dropna(), patch_artist=True,
                               boxprops=dict(facecolor="#0f4c81", color="#38bdf8"),
                               medianprops=dict(color="#f59e0b", linewidth=2),
                               whiskerprops=dict(color="#38bdf8"),
                               capprops=dict(color="#38bdf8"),
                               flierprops=dict(markerfacecolor="#f43f5e", marker="o", markersize=4))
                    ax.set_title(col, fontsize=10)
                    ax.grid(True, alpha=0.3)
                    r_cols[idx].pyplot(fig, use_container_width=True)
                    plt.close()

            section("Automated Insights")
            with st.expander("View Insights", expanded=True):
                corr_vals = corr.abs().unstack()
                corr_vals = corr_vals[corr_vals < 1].sort_values(ascending=False)
                if not corr_vals.empty:
                    top = corr_vals.index[0]
                    st.markdown(f"<div class='insight-box'>🔗 <b>Strongest correlation:</b> <code>{top[0]}</code> and <code>{top[1]}</code> — r = {corr_vals.iloc[0]:.3f}</div>", unsafe_allow_html=True)

                top_std = num_df.std().idxmax()
                st.markdown(f"<div class='insight-box'>📊 <b>Highest variation:</b> <code>{top_std}</code> (Std Dev = {num_df[top_std].std():.3f})</div>", unsafe_allow_html=True)

                outlier_cols = []
                for col in num_df.columns:
                    q1, q3 = num_df[col].quantile(0.25), num_df[col].quantile(0.75)
                    iqr = q3 - q1
                    n_out = ((num_df[col] < q1 - 1.5 * iqr) | (num_df[col] > q3 + 1.5 * iqr)).sum()
                    if n_out > 0:
                        outlier_cols.append(f"{col} ({n_out})")
                if outlier_cols:
                    st.markdown(f"<div class='insight-box'>⚠️ <b>Potential outliers detected in:</b> {', '.join(outlier_cols)}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='insight-box'>✅ <b>No significant outliers detected.</b></div>", unsafe_allow_html=True)

elif page == "📈 Visualizations":
    hero("Data Visualizations", "Generate custom charts to communicate your findings clearly.")
    df = get_df()
    if df is None:
        no_data_msg()
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        chart_type = st.selectbox("Select Chart Type", [
            "Histogram", "Bar Chart", "Scatter Plot",
            "Line Chart", "Box Plot", "Correlation Heatmap"
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        fig = None

        if chart_type == "Histogram":
            col = st.selectbox("Select Numeric Column", num_cols)
            bins = st.slider("Number of Bins", 5, 80, 20)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df[col].dropna(), bins=bins, color="#38bdf8", edgecolor="#0d0f1a", alpha=0.85)
            ax.set_title(f"Histogram — {col}", fontsize=14)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        elif chart_type == "Bar Chart":
            col = st.selectbox("Select Column", all_cols)
            top_n = st.slider("Top N values", 5, 30, 10)
            counts = df[col].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(counts.index.astype(str), counts.values,
                   color=PALETTE[:len(counts)], edgecolor="#0d0f1a")
            ax.set_title(f"Bar Chart — {col} (Top {top_n})", fontsize=14)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            ax.grid(True, alpha=0.3)

        elif chart_type == "Scatter Plot":
            xc = st.selectbox("X Axis", num_cols, index=0)
            yc = st.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(df[xc], df[yc], alpha=0.6, color="#38bdf8",
                       edgecolors="#0d7377", linewidths=0.4, s=40)
            ax.set_title(f"Scatter Plot — {xc} vs {yc}", fontsize=14)
            ax.set_xlabel(xc)
            ax.set_ylabel(yc)
            ax.grid(True, alpha=0.3)

        elif chart_type == "Line Chart":
            col = st.selectbox("Select Numeric Column", num_cols)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df[col].values, color="#38bdf8", linewidth=1.5, alpha=0.85)
            ax.fill_between(range(len(df[col])), df[col].values, alpha=0.15, color="#0d7377")
            ax.set_title(f"Line Chart — {col}", fontsize=14)
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

        elif chart_type == "Box Plot":
            col = st.selectbox("Select Numeric Column", num_cols)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot(df[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor="#0f4c81", color="#38bdf8"),
                       medianprops=dict(color="#f59e0b", linewidth=2),
                       whiskerprops=dict(color="#38bdf8"),
                       capprops=dict(color="#38bdf8"),
                       flierprops=dict(markerfacecolor="#f43f5e", marker="o", markersize=5))
            ax.set_title(f"Box Plot — {col}", fontsize=14)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

        elif chart_type == "Correlation Heatmap":
            num_df = df.select_dtypes(include=np.number)
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", ax=ax,
                        cmap="coolwarm", linewidths=0.5,
                        annot_kws={"size": 8},
                        cbar_kws={"shrink": 0.8})
            ax.set_title("Correlation Heatmap", fontsize=14, pad=12)

        if fig:
            fig.patch.set_facecolor("#141827")
            st.pyplot(fig, use_container_width=True)
            plt.close()

            buf2 = io.BytesIO()
            fig.savefig(buf2, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#141827")
            buf2.seek(0)
            st.download_button("⬇️ Download Chart", buf2,
                               f"{chart_type.lower().replace(' ', '_')}.png", "image/png")

elif page == "🤖 Prediction":
    hero("Predictive Modelling", "Train a Linear Regression model and evaluate its performance.")
    df = get_df()
    if df is None:
        no_data_msg()
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for prediction.")
        else:
            target = st.selectbox("Select Target Column (Y)", num_cols)
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)

            if st.button("🚀 Train Model"):
                features = [c for c in num_cols if c != target]
                X = df[features].values
                y = df[target].values

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size / 100, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                c1, c2, c3, c4 = st.columns(4)
                metric_card(c1, f"{mae:.4f}", "MAE")
                metric_card(c2, f"{mse:.4f}", "MSE")
                metric_card(c3, f"{rmse:.4f}", "RMSE")
                metric_card(c4, f"{r2:.4f}", "R² Score")

                st.markdown("<br>", unsafe_allow_html=True)

                if r2 >= 0.85:
                    st.success(f"🏆 Excellent model performance! R² = {r2:.4f} — Strong predictive power.")
                elif r2 >= 0.60:
                    st.info(f"📊 Moderate predictive power. R² = {r2:.4f} — Model captures general trends.")
                else:
                    st.warning(f"⚠️ Improvement needed. R² = {r2:.4f} — Consider feature engineering or different algorithms.")

                section("Actual vs Predicted Values")
                sample_n = min(100, len(y_test))
                idxs = np.random.choice(len(y_test), sample_n, replace=False)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(y_test[idxs], color="#38bdf8", linewidth=1.5, label="Actual", alpha=0.85)
                ax.plot(y_pred[idxs], color="#f59e0b", linewidth=1.5, label="Predicted", alpha=0.85, linestyle="--")
                ax.fill_between(range(sample_n), y_test[idxs], y_pred[idxs],
                                alpha=0.07, color="#f43f5e")
                ax.set_title(f"Actual vs Predicted — {target}", fontsize=13)
                ax.set_xlabel("Sample Index")
                ax.set_ylabel(target)
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)
                fig.patch.set_facecolor("#141827")
                st.pyplot(fig, use_container_width=True)
                plt.close()

                section("Regression Fit Plot")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                ax2.scatter(y_test, y_pred, alpha=0.5, color="#38bdf8",
                            edgecolors="#0d7377", linewidths=0.4, s=35)
                mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                ax2.plot([mn, mx], [mn, mx], color="#f59e0b", linewidth=2, linestyle="--", label="Perfect Fit")
                ax2.set_title("Actual vs Predicted — Regression Fit", fontsize=13)
                ax2.set_xlabel("Actual")
                ax2.set_ylabel("Predicted")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                fig2.patch.set_facecolor("#141827")
                st.pyplot(fig2, use_container_width=True)
                plt.close()

                with st.expander("📌 Model Coefficients (Feature Importance)"):
                    coef_df = pd.DataFrame({
                        "Feature": features,
                        "Coefficient": model.coef_
                    }).sort_values("Coefficient", ascending=False)
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)
