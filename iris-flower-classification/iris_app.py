import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_knn_model.pkl")

# App UI
st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower measurements below to predict the species:")

# User input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"🌼 Predicted species: **{prediction.capitalize()}**")
