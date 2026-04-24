import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# ---- CONFIG ----
st.set_page_config(page_title="Churn Predictor", layout="wide")

# ---- LOAD MODEL ----
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---- CUSTOM CSS ----
st.markdown("""
<style>
.main {
    padding: 2rem 4rem;
}
.block-container {
    max-width: 1100px;
    margin: auto;
}
.section {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.title("🚀 Customer Churn Prediction System")
st.caption("Advanced ML-powered churn prediction")

st.markdown("---")

# =========================
# CUSTOMER INFO
# =========================
st.markdown("## 👤 Customer Info")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])

with col2:
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

# =========================
# SERVICES
# =========================
st.markdown("## 📡 Services")

col3, col4 = st.columns(2)

with col3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes"])
    backup = st.selectbox("Online Backup", ["No", "Yes"])

with col4:
    protection = st.selectbox("Device Protection", ["No", "Yes"])
    support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming = st.selectbox("Streaming (TV + Movies)", ["No", "Yes"])

# =========================
# BILLING
# =========================
st.markdown("## 💳 Billing")

col5, col6 = st.columns(2)

with col5:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

with col6:
    monthly = st.number_input("Monthly Charges", value=70.0)
    total = st.number_input("Total Charges", value=1000.0)

st.markdown("---")

# =========================
# PREDICT
# =========================
if st.button("🔮 Predict Churn"):

    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming,
        "StreamingMovies": streaming,
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.markdown("## 🎯 Result")

    # ---- GAUGE ----
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if prediction else "green"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---- RESULT TEXT ----
    if prediction == 1:
        st.error("⚠️ High Risk: Customer likely to churn")
    else:
        st.success("✅ Low Risk: Customer likely to stay")