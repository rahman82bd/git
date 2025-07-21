# app.py

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Streamlit App
st.title("🌼 Iris Flower Species Classifier")
st.write("This web app predicts the species of an Iris flower based on user input.")

# User input sliders
sepal_length = st.slider("Sepal length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width  = st.slider("Sepal width (cm)",  float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.slider("Petal length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width  = st.slider("Petal width (cm)",  float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

# Predict
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)
input_scaled = scaler.transform(input_data)
prediction = clf.predict(input_scaled)
predicted_class = target_names[prediction[0]]

st.subheader("Prediction")
st.success(f"The predicted Iris species is **{predicted_class}**.")
