import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

st.title("Task 5 - Predictive Modeling")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    df = df.select_dtypes(include=np.number)

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    st.write("MAE:", mean_absolute_error(y_test, predictions))
    st.write("MSE:", mean_squared_error(y_test, predictions))
    st.write("R2 Score:", r2_score(y_test, predictions))