
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f9fc;
            color: #14213d;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-card, .info-card, .result-card, .metric-card {
            background: #ffffff;
            border: 1px solid rgba(20, 33, 61, 0.08);
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(20, 33, 61, 0.06);
        }

        .hero-card {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
            color: #102542;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #5b6475;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #102542;
            margin: 0.25rem 0 0.15rem 0;
        }

        .section-subtitle {
            color: #5b6475;
            margin-bottom: 0.75rem;
        }

        .result-card {
            padding: 1.25rem 1.4rem;
            margin: 0.9rem 0 1rem 0;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            text-align: center;
        }

        .metric-label {
            color: #5b6475;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: #102542;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0;
        }

        .sidebar-shell {
            padding: 1rem 0.9rem;
        }

        .sidebar-brand {
            font-size: 1.15rem;
            font-weight: 800;
            color: #102542;
            margin-bottom: 0.25rem;
        }

        .sidebar-note {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .sidebar-footer {
            position: sticky;
            top: 85vh;
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(20, 33, 61, 0.08);
            color: #64748b;
            font-size: 0.82rem;
        }

        .feature-list li, .tech-list li {
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer():
    st.markdown(
        """
        <div class="sidebar-footer">
            Built with Streamlit for a clean loan screening workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_model():
    try:
        return joblib.load("models/loan_approval_pipeline.joblib")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

st.sidebar.markdown(
    """
    <div class="sidebar-shell">
        <div class="sidebar-brand">🏦 Loan Approval Prediction</div>
        <div class="sidebar-note">Professional loan screening dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Navigation", ["Home","Model Information","About"])
render_sidebar_footer()

if page=="Home":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🏦 Loan Approval Prediction</div>
            <p class="hero-subtitle">Predict whether a loan application is likely to be approved using a trained Logistic Regression model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="section-title">📝 Applicant Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter the applicant details below to evaluate the loan decision.</div>', unsafe_allow_html=True)
    col1,col2=st.columns(2, gap="large")
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

    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    with button_col2:
        predict_clicked = st.button("🔍 Predict Loan Status", use_container_width=True, type="primary")

    if predict_clicked:
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
        st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)
        if pred==1:
            st.markdown(
                """
                <div class="result-card" style="border-left: 6px solid #16a34a;">
                    <h3 style="margin: 0; color: #166534;">✅ Loan Approved</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            conf=proba[1]
            decision_text = "The model predicts that this applicant is likely to receive loan approval based on the provided information."
        else:
            st.markdown(
                """
                <div class="result-card" style="border-left: 6px solid #dc2626;">
                    <h3 style="margin: 0; color: #991b1b;">❌ Loan Rejected</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            conf=proba[0]
            decision_text = "The model predicts that this applicant may not qualify for loan approval based on the provided information."

        st.markdown(
            f"""
            <div class="result-card">
                <p style="margin: 0; color: #334155; font-size: 0.98rem;">{decision_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1,c2,c3 = st.columns(3, gap="medium")
        with c1:
            render_metric_card("Decision", "Approved" if pred==1 else "Rejected")
        with c2:
            render_metric_card("Approval Probability", f"{proba[1]*100:.2f}%")
        with c3:
            render_metric_card("Rejection Probability", f"{proba[0]*100:.2f}%")

        st.markdown('<div class="section-title">Confidence Progress Bar</div>', unsafe_allow_html=True)
        st.progress(float(conf))
        st.caption(f"Model Confidence: {conf*100:.2f}%")

elif page=="Model Information":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">📊 Model Information</div>
            <p class="hero-subtitle">A concise summary of the model pipeline and input features used for predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Algorithm</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card"><p style="margin: 0;">Logistic Regression</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-card"><p style="margin: 0;">ColumnTransformer<br>StandardScaler<br>OneHotEncoder<br>LogisticRegression</p></div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">Features Used</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="result-card">
                <ul class="feature-list" style="margin: 0; padding-left: 1.2rem; color: #334155;">
                    <li>person_age</li>
                    <li>person_gender</li>
                    <li>person_income</li>
                    <li>loan_amnt</li>
                    <li>loan_intent</li>
                    <li>loan_percent_income</li>
                    <li>credit_score</li>
                    <li>previous_loan_defaults_on_file</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">ℹ️ About</div>
            <p class="hero-subtitle">A streamlined loan prediction dashboard designed for quick, interpretable screening.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    overview_col, detail_col = st.columns([1, 1], gap="large")
    with overview_col:
        st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card"><p style="margin: 0;">This application helps estimate whether a loan application is likely to be approved or rejected based on applicant information and a trained machine learning pipeline.</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Purpose</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card"><p style="margin: 0;">To provide an intuitive loan screening experience with a clean, decision-focused user interface.</p></div>', unsafe_allow_html=True)

    with detail_col:
        st.markdown('<div class="section-title">Technologies Used</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="result-card">
                <ul class="tech-list" style="margin: 0; padding-left: 1.2rem; color: #334155;">
                    <li>Python</li>
                    <li>Streamlit</li>
                    <li>Pandas</li>
                    <li>Scikit-learn</li>
                    <li>Joblib</li>
                    <li>GitHub Codespaces</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card"><p style="margin: 0;">Loan Approval Prediction Dataset</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Developer</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card"><p style="margin: 0; color: #64748b;">Developer information coming soon.</p></div>', unsafe_allow_html=True)
