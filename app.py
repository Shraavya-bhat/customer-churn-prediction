import streamlit as st
import pickle
import pandas as pd

# ---- CONFIG ----
st.set_page_config(page_title="Churn Predictor", layout="wide")

# ---- LOAD ----
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---- STYLE ----
st.markdown("""
<style>
.main {background-color: #0E1117;}
h1 {text-align: center;}
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.title("🚀 Customer Churn Prediction System")

# ---- LAYOUT ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 0, 72)

with col2:
    st.subheader("📡 Service Details")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.subheader("💰 Billing Info")
col3, col4 = st.columns(2)
with col3:
    MonthlyCharges = st.number_input("Monthly Charges", value=50.0)
with col4:
    TotalCharges = st.number_input("Total Charges", value=500.0)

# ---- INPUT ----
input_dict = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": Contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

# ---- PREDICT ----
if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    
    if prediction == 1:
        st.markdown(
            f"<div class='result-box' style='background:#ff4b4b;'>⚠️ HIGH CHURN RISK ({probability:.2%})</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-box' style='background:#00c853;'>✅ LOW CHURN RISK ({probability:.2%})</div>",
            unsafe_allow_html=True
        )

    # ---- PROGRESS BAR ----
    st.progress(float(probability))

    # ---- SUGGESTIONS ----
    st.subheader("💡 Suggested Action")

    if probability > 0.6:
        st.warning("Offer discount or retention plan immediately!")
    elif probability > 0.4:
        st.info("Engage customer with offers or support.")
    else:
        st.success("Customer is stable. Maintain service quality.")