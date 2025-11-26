import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pickle

# loading data
with open('model.pkl', 'rb') as file:
    rf = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('columns.pkl', 'rb') as file:
    all_columns = pickle.load(file)

with open('numeric_cols.pkl', 'rb') as file:
    numeric_cols = pickle.load(file)


# page configuration
st.set_page_config(page_title="Financial Risk Assessment", layout="wide")
st.title("Financial Risk Assessment Application")

st.write("""
This application assesses the financial risk of individuals based on their financial data.
""")
st.write("Please input the required financial details below:")
with st.form("risk_form"):
    st.subheader("Input Financial Details")
    year = st.number_input("Year", min_value=2000, max_value=2030)
    household_size = st.number_input("Household Size", min_value=1, max_value=20)
    age_of_respondent = st.number_input("Respondent Age", min_value=18, max_value=100)
    submit_button = st.form_submit_button("Assess Risk")

if 'year' in locals() and submit_button:
    input_data = pd.DataFrame({
        'year': [year],
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent]
    })

    input_data[scaler.feature_names_in_] = scaler.transform(
    input_data[scaler.feature_names_in_]
)
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=all_columns, fill_value=0)

    # predicting risk
    risk_prediction = rf.predict(input_data_encoded)
    risk_probability = rf.predict_proba(input_data_encoded)[:, 1][0]
    # displaying results
    st.subheader("Risk Assessment Result")
    if risk_prediction[0] == 1:
        st.error(f"The individual is assessed to be at HIGH financial risk with a probability of {risk_probability:.2%}.")
    else:
        st.success(f"The individual is assessed to be at LOW financial risk with a probability of {1 - risk_probability:.2%}.")
    # visualizing risk probability
    fig, ax = plt.subplots()
    sns.barplot(x=['Low Risk', 'High Risk'], y=[1 - risk_probability, risk_probability], ax=ax)
    ax.set_ylabel("Probability")
    ax.set_title("Financial Risk Probability")
    st.pyplot(fig)
st.write("""
---
Developed by Amins Analytics Team.
© 2024 Amins Analytics. All rights reserved.
""")
    

