import streamlit as st
import pandas as pd
import numpy as np

st.title("Task 3 - Exploratory Data Analysis")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    numeric_df = df.select_dtypes(include=np.number)

    st.subheader("Basic Statistics")

    stats = numeric_df.agg(
        ["mean", "median", "std", "min", "max"]
    ).T

    st.dataframe(stats)

    st.subheader("Correlation Matrix")
    st.dataframe(numeric_df.corr())

    st.subheader("Outlier Detection")

    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)

        iqr = q3 - q1

        outliers = (
            (numeric_df[col] < q1 - 1.5 * iqr)
            |
            (numeric_df[col] > q3 + 1.5 * iqr)
        ).sum()

        st.write(f"{col}: {outliers} outliers")