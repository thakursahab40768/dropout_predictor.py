import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Early Dropout Prediction System")

model = joblib.load("dropout_model.pkl")

gpa = st.slider("GPA", 0.0, 10.0, 5.0)
attendance = st.slider("Attendance Rate", 0.0, 1.0, 0.75)
assignments = st.slider("Assignment Completion Rate", 0.0, 1.0, 0.80)
login_freq = st.number_input("Average Weekly Logins", 0, 50, 5)

if st.button("Predict Dropout Risk"):
    input_df = pd.DataFrame([[gpa, attendance, assignments, login_freq]],
                            columns=['gpa', 'attendance_rate', 'assignment_completion', 'login_frequency'])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Dropout Risk (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Low Dropout Risk (Confidence: {1 - probability:.2f})")
        
