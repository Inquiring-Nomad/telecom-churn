import pandas as pd
import streamlit as st
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV, \
    cross_val_predict
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, \
    accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-white')


class CustomBinaryTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.replace(to_replace={'Yes': 1, 'No': 0})


model_pickle_in = open("logReg_churn_model.pkl", "rb")
prep_pickle_in = open("prepro_churn_model.pkl", "rb")
model = pickle.load(model_pickle_in)
prepr = pickle.load(prep_pickle_in)
st.title("Churn Predictions")


def predict_churn(gender, senior, partner, dependents, tenure, phoneservice, multiplelines, internet, onlinesec,
                  onlinebac, deviceprot, techsup, streamtv, streammov, contract, paperless, paymentmeth,
                  monthlycharges, totalcharges):
    df = pd.DataFrame(
        data=[[gender, senior, partner, dependents, tenure, phoneservice, multiplelines, internet, onlinesec,
         onlinebac, deviceprot, techsup, streamtv, streammov, contract, paperless, paymentmeth,
         monthlycharges, totalcharges]],
        columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity',
                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                 'StreamingMovies', 'Contract', 'PaperlessBilling',
                 'PaymentMethod',
                 'MonthlyCharges', 'TotalCharges'])
    X = prepr.transform(df)

    y = model.predict(X)
    return y
    # 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    # 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    # 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    # 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    # 'MonthlyCharges', 'TotalCharges'
    pass


@st.cache
def load_data(dataset):
    return pd.read_csv(dataset)


def main():
    data = load_data('data/Telecomms-Churn.csv')
    binary_vars = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    categ_vars = ['gender', 'MultipleLines',
                  'InternetService',
                  'OnlineSecurity',
                  'OnlineBackup',
                  'DeviceProtection',
                  'TechSupport',
                  'StreamingTV',
                  'StreamingMovies',
                  'Contract',
                  'PaymentMethod']

    numeric_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']

    with st.form(key='churn-predict'):
        gender = st.selectbox("Gender", data['gender'].unique())
        senior = st.selectbox("Senior Citizen", data['SeniorCitizen'].unique(),
                              format_func=lambda x: 'Yes' if x == 1 else 'No')
        partner = st.selectbox("Has a partner", data['Partner'].unique())
        dependents = st.selectbox("Has Dependents", data['Dependents'].unique())
        tenure = st.number_input("Tenure", min_value=1)
        phoneservice = st.selectbox("Has phone service", data['PhoneService'].unique())
        multiplelines = st.selectbox("Multiple Lines?", data['MultipleLines'].unique())
        internet = st.selectbox("Internet Service", data['InternetService'].unique())

        onlinesec = st.selectbox("Online Security", data['OnlineSecurity'].unique())
        onlinebac = st.selectbox("Online Backup", data['OnlineBackup'].unique())
        deviceprot = st.selectbox("Device Protection", data['DeviceProtection'].unique())
        techsup = st.selectbox("Tech Support", data['TechSupport'].unique())
        streamtv = st.selectbox("Streaming TV", data['StreamingTV'].unique())
        streammov = st.selectbox("Streaming Movies", data['StreamingMovies'].unique())
        contract = st.selectbox("Contract", data['Contract'].unique())

        paperless = st.selectbox("Paperless Billing", data['PaperlessBilling'].unique())
        paymentmeth = st.selectbox("Payment Method", data['PaymentMethod'].unique())

        monthlycharges = st.number_input("Monthly Charges", min_value=1)
        totalcharges = st.number_input("Total Charges", min_value=1)
        submit_button = st.form_submit_button(label='Predict')
        if submit_button:
            pred = predict_churn(gender, senior, partner, dependents, tenure, phoneservice, multiplelines, internet,
                                 onlinesec,
                                 onlinebac, deviceprot, techsup, streamtv, streammov, contract, paperless, paymentmeth,
                                 monthlycharges, totalcharges)
            if(pred == 0):
                st.success("There is a high probability that the customer will keep using your services.")
            else:
                st.error("There is a high probability that the customer will stop using your services.")

    # result = ""
    # if st.button("Predict"):
    #     result = predict_churn()
    # st.success('The output is {}'.format(result))


if __name__ == '__main__':
    main()
