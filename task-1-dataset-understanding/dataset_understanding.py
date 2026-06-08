import streamlit as st
import pandas as pd
import numpy as np

st.title("Task 1 - Dataset Understanding")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Size")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.reset_index())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.subheader("Features")
    st.write("Numeric Columns:", numeric_cols)
    st.write("Categorical Columns:", categorical_cols)

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().reset_index())

    st.subheader("Dataset Summary")
    st.dataframe(df.describe(include="all"))