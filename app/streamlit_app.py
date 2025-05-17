import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/logistic_model.joblib")
scaler = joblib.load("models/scaler.joblib")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger's data below to predict the probability of survival.")

sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
age = st.slider("Age", 0, 100, 30)
sibsp = st.slider("Number of siblings/spouses aboard", 0, 8, 0)
parch = st.slider("Number of parents/children aboard", 0, 6, 0)
fare = st.slider("Fare (ticket price)", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

sex_num = 0 if sex == "male" else 1
embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0

X = np.array([[pclass, sex_num, age, sibsp, parch, fare, embarked_c, embarked_q]])
X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)[0]
probability = model.predict_proba(X_scaled)[0][1]

st.subheader("🧠 Model Prediction")
st.write(f"**Survived:** {'Yes' if prediction == 1 else 'No'}")
st.write(f"**Survival Probability:** {probability:.2%}")
