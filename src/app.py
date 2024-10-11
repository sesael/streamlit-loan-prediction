"""
The Render web link for the project is:

streamlit-loan-prediction.onrender.com

"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Loading the saved model and preprocessors
with open('models/decision_tree_model.pkl', 'rb') as f:
    loaded_dtc = pickle.load(f)

with open('models/one_hot_encoder.pkl', 'rb') as f:
    loaded_ohe = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    loaded_le = pickle.load(f)

# Defining the categorical and numerical columns
categorical_oh_cols = [
    'person_home_ownership',
    'loan_intent',
    'cb_person_default_on_file'
]

categorical_le_cols = ["loan_grade"]

numerical_cols = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
]

# Streamlit app
def main():
    st.title("Loan Approval Prediction")
    st.write("Enter the details below to predict if the loan will be approved or denied.")

    # Input fields for numerical features
    person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Person Income", min_value=0.0, value=50000.0, step=1000.0)
    person_emp_length = st.slider("Person Employment Length (years)", min_value=0.0, value=5.0, step=0.5)
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    cb_person_cred_hist_length = st.slider("Credit History Length", min_value=0, max_value=50, value=10)

    # Input fields for categorical features
    person_home_ownership = st.selectbox(
        "Person Home Ownership",
        options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    )

    cb_person_default_on_file = st.selectbox(
        "Default on File",
        options=['Y', 'N'],
    )

    # Creating a DataFrame from the input
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_emp_length': [person_emp_length],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'person_home_ownership': [person_home_ownership],
        'loan_intent': [loan_intent],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'loan_grade': [loan_grade],
    })

    # Preprocessing the input data
    # Separate numerical and categorical data
    X_new_num = input_data[numerical_cols]
    X_new_oh_cat = input_data[categorical_oh_cols]

    # Reindex to ensure all expected columns from the OneHotEncoder are present
    #X_new_oh_cat = X_new_oh_cat.reindex(columns=loaded_ohe.feature_names_in_, fill_value=0)

    # Transform categorical features using OneHotEncoder (convert sparse matrix to dense)
    X_new_cat_oh_encoded = loaded_ohe.transform(X_new_oh_cat).toarray()  # Convert sparse matrix to dense

    # Check the transformed data
    print("Transformed OneHotEncoder shape:", X_new_cat_oh_encoded.shape)
    encoded_column_names = loaded_ohe.get_feature_names_out()
    print("Expected columns:", encoded_column_names)

    # Ensure the shape matches
    X_new_cat_oh_encoded_df = pd.DataFrame(X_new_cat_oh_encoded, columns=encoded_column_names)

    # Handling 'loan_grade' separately (using LabelEncoder)
    X_new_loan_grade = input_data[['loan_grade']]
    X_new_loan_grade_encoded = loaded_le.transform(X_new_loan_grade['loan_grade'])
    X_new_loan_grade_encoded_df = pd.DataFrame(
        X_new_loan_grade_encoded,
        columns=['loan_grade']
    )

    # Combine processed features
    X_new_cat_encoded = pd.concat([X_new_cat_oh_encoded_df, X_new_loan_grade_encoded_df], axis=1)
    X_new_processed = pd.concat([X_new_num, X_new_cat_encoded], axis=1)

    #st.write("Processed Input Data for Prediction", X_new_processed)

    # Prediction
    if st.button("Predict"):
        #st.write("Processed Input Data for Prediction:", X_new_processed)
        
        prediction = loaded_dtc.predict(X_new_processed)
        probability = loaded_dtc.predict_proba(X_new_processed)[0]
        
        # Apply probability smoothing
        epsilon = 1e-3  # Small value to prevent 0 or 1
        smoothed_prob = [max(min(probability[0], 1 - epsilon), epsilon),
                        max(min(probability[1], 1 - epsilon), epsilon)]
        
        #st.write("Model Prediction:", prediction[0])
        #st.write("Model Probabilities (Denial, Approval):", smoothed_prob)

        if prediction[0] == 1:
            st.success(f"The loan is likely to be **Denied** with a probability of {smoothed_prob[1]:.2%}.")
        else:
            st.error(f"The loan is likely to be **Approval** with a probability of {smoothed_prob[0]:.2%}.")


if __name__ == '__main__':
    main()
