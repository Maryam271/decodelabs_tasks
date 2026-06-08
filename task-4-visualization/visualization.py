import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Task 4 - Data Visualization")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    numeric_cols = df.select_dtypes(include=np.number).columns

    col = st.selectbox("Select Column", numeric_cols)

    fig, ax = plt.subplots()
    ax.hist(df[col].dropna())
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.boxplot(df[col].dropna())
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax3)
    st.pyplot(fig3)