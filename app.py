import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# ---- CONFIG ----
st.set_page_config(page_title="Churn Predictor", layout="wide")

# ---- LOAD MODEL ----
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---- CSS ----
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 2rem;
}
.section {
    background: #0f172a;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
}
.stButton button {
    width: 220px;
    height: 45px;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.title("🚀 Customer Churn Prediction System")
st.caption("This app predicts whether a customer is likely to churn based on their service usage and billing behavior.")

st.markdown("---")

# =========================
# SAMPLE AUTO-FILL
# =========================
if st.button("⚡ Use Sample Customer"):
    st.session_state.sample = True
else:
    st.session_state.sample = False

# =========================
# BASIC INPUTS (SIMPLIFIED)
# =========================
st.markdown("## 🎯 Key Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12 if not st.session_state.sample else 5)

with col2:
    monthly = st.slider("Monthly Charges", 0, 150, 70 if not st.session_state.sample else 95)

with col3:
    contract = st.selectbox("Contract Type",
        ["Month-to-month", "One year", "Two year"],
        index=0 if not st.session_state.sample else 0
    )

# =========================
# ADVANCED OPTIONS
# =========================
with st.expander("⚙️ Advanced Options"):

    col4, col5 = st.columns(2)

    with col4:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col5:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes"])
        support = st.selectbox("Tech Support", ["No", "Yes"])
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    total = st.slider("Total Charges", 0, 5000, 1000 if not st.session_state.sample else 300)

# =========================
# PREDICT
# =========================
st.markdown("<br>", unsafe_allow_html=True)

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
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
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

    # =========================
    # RISK LEVEL
    # =========================
    if prob > 0.7:
        risk = "High Risk ⚠️"
        color = "red"
    elif prob > 0.4:
        risk = "Medium Risk 🟡"
        color = "orange"
    else:
        risk = "Low Risk ✅"
        color = "green"

    # =========================
    # GAUGE
    # =========================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### {risk} ({prob*100:.2f}%)")

    # =========================
    # REASONS (SMART LOGIC)
    # =========================
    reasons = []

    if contract == "Month-to-month":
        reasons.append("Month-to-month contract")
    if monthly > 80:
        reasons.append("High monthly charges")
    if support == "No":
        reasons.append("No tech support")
    if tenure < 6:
        reasons.append("New customer")

    if reasons:
        st.markdown("### 🧠 Why this prediction?")
        for r in reasons:
            st.write("•", r)

    # =========================
    # BUSINESS SUGGESTION
    # =========================
    st.markdown("### 💡 What should the company do?")

    if prob > 0.7:
        st.error("Offer discounts / retention calls immediately")
    elif prob > 0.4:
        st.warning("Engage customer with offers & support")
    else:
        st.success("No action needed — customer is stable")