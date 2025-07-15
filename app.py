import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the following health details to check diabetes risk:")

# Input fields with range hints
preg = st.number_input("Pregnancies", min_value=0, help="Typical range: 0–17")
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, help="Normal: 70–140 mg/dL")
bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, help="Normal: ~80 mm Hg (diastolic)")
skin = st.number_input("Skin Thickness (mm)", min_value=0, help="Typical range: 10–50 mm")
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, help="Normal: 15–276 mu U/ml")
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, format="%.1f", help="Normal: 18.5–24.9")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", help="Relative risk: 0.0–2.5+")
age = st.number_input("Age (years)", min_value=0, help="Typical range: 18–90")

# Predict button
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("🚨 High risk of diabetes detected.")
    else:
        st.success("✅ Low risk of diabetes.")
