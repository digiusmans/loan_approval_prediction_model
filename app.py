
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    try:
        return joblib.load("models/loan_approval_pipeline.joblib")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

st.sidebar.title("🏦 Loan Approval")
page = st.sidebar.radio("Navigation", ["Home","Model Information","About"])

if page=="Home":
    st.markdown("""
# 🏦 Loan Approval Prediction

Predict whether a loan application is likely to be **Approved** or **Rejected**
using a Machine Learning model trained on historical loan applicant data.

---
""")
    st.write("Fill in the applicant information below.")
    st.subheader("📝 Applicant Information")
    col1,col2=st.columns(2)
    with col1:
        person_age=st.number_input("Age",18,100,25)
        person_gender=st.selectbox("Gender",["female","male"])
        person_income=st.number_input("Annual Income",min_value=0.0,value=50000.0,step=1000.0)
        loan_amnt=st.number_input("Loan Amount",min_value=500.0,value=10000.0,step=500.0)
    with col2:
        loan_intent=st.selectbox("Loan Purpose",["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
        loan_percent_income=st.slider("Loan Percent Income",0.0,1.0,0.20,0.01)
        credit_score=st.slider("Credit Score",300,850,650)
        previous_loan_defaults_on_file=st.selectbox("Previous Loan Default",["No","Yes"])

    if st.button("Predict Loan Status", use_container_width=True):
        input_df=pd.DataFrame({
            "person_age":[person_age],
            "person_gender":[person_gender],
            "person_income":[person_income],
            "loan_amnt":[loan_amnt],
            "loan_intent":[loan_intent],
            "loan_percent_income":[loan_percent_income],
            "credit_score":[credit_score],
            "previous_loan_defaults_on_file":[previous_loan_defaults_on_file]
        })
        pred=model.predict(input_df)[0]
        proba=model.predict_proba(input_df)[0]
        st.divider()
        if pred==1:
            st.success("✅ Loan Approved")
            conf=proba[1]
        else:
            st.error("❌ Loan Rejected")
            conf=proba[0]
        c1,c2=st.columns(2)
        c1.metric("Approval Probability",f"{proba[1]*100:.2f}%")
        c2.metric("Rejection Probability",f"{proba[0]*100:.2f}%")
        st.progress(float(conf))
        st.caption(f"Prediction confidence: {conf*100:.2f}%")

elif page=="Model Information":
    st.title("📊 Model Information")
    st.write("- Algorithm: Logistic Regression")
    st.write("- Pipeline: ColumnTransformer + StandardScaler + OneHotEncoder + LogisticRegression")
    st.write("- Features: 8")
    st.write("- Saved using Joblib")

else:
    st.title("ℹ️ About")
    st.write("Loan Approval Prediction project built with Streamlit and scikit-learn.")
    st.write("Technologies: Python, Pandas, Scikit-learn, Joblib, Streamlit, GitHub Codespaces.")
