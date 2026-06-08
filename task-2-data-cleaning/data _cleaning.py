import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("Task 2 - Data Cleaning & Preprocessing")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.write("Original Shape:", df.shape)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    df.drop_duplicates(inplace=True)

    le = LabelEncoder()

    for col in df.select_dtypes(exclude=np.number).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    st.success("Data Cleaned Successfully")

    st.dataframe(df.head())

    st.download_button(
        "Download Clean Dataset",
        df.to_csv(index=False),
        "cleaned_dataset.csv"
    )