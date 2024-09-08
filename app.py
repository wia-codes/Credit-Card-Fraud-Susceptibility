import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import load_model, predict_susceptibility

# Load the trained model
model = load_model()

# Title for the app
st.title("Credit Card Transactions Fraud Detection")

# Description of the app
st.write("""
This app predicts whether a person is highly susceptible or low susceptible to fraud based on their transaction amount, age, and gender.
""")

# Create input fields for user inputs
amt = st.number_input("Transaction Amount", min_value=0, max_value=10000, value=500)
age = st.slider("Age", min_value=18, max_value=100, value=35)
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode Gender (0 for Female, 1 for Male)
gender_encoded = 1 if gender == "Male" else 0

# Prediction Button
if st.button("Predict Fraud Susceptibility"):
    result = predict_susceptibility(model, amt, age, gender_encoded)
    st.write(f"Prediction: {result}")

# Load the CSV file for visualizations
data = pd.read_csv('fraudTest.csv')

# Process data for visualizations

# Convert 'trans_date_trans_time' and 'dob' to datetime
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])

# Create 'age' column
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year

# Bar Chart: Fraud Count by Age Group
st.subheader("Fraud Count by Age Group")
data['age_group'] = pd.cut(data['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
age_group_fraud = data.groupby('age_group')['is_fraud'].sum()
st.bar_chart(age_group_fraud)

# Bar Chart: Fraud Count by Gender
st.subheader("Fraud Count by Gender")
gender_fraud = data.groupby('gender')['is_fraud'].sum()
gender_fraud.index = ['Female', 'Male']  
st.bar_chart(gender_fraud)