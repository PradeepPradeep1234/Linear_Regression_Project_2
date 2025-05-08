import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("Modal Price Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/daily_price.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Correlation Heatmap
if st.checkbox("Show Correlation Heatmap"):
    numeric_df = df[["Min Price", "Max Price", "Modal Price"]]
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Pairplot
if st.checkbox("Show Pairplot"):
    st.write("This might take a few seconds...")
    st.pyplot(sns.pairplot(df[["Min Price", "Max Price", "Modal Price"]]).fig)

# Feature selection
X = df.drop(columns=["Modal Price"])
y = df["Modal Price"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing and model
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"**MAE**: {mae:.2f}")
st.write(f"**RMSE**: {rmse:.2f}")
st.write(f"**R² Score**: {r2:.4f}")

# Scatter plot
st.subheader("Actual vs Predicted Modal Prices")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.5, color='blue')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Actual vs Predicted Modal Prices")
st.pyplot(fig2)
