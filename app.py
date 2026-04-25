import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🚀 Customer Churn Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Predict churn risk + view insights")

st.markdown("---")

# ---------------- SAMPLE DATA (for charts) ----------------
# (Use your dataset here if you have one)
@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    return df

df = load_data()

# ---------------- KPI SECTION ----------------
col1, col2, col3, col4 = st.columns(4)

churn_rate = (df['Churn'] == 'Yes').mean() * 100
avg_charge = df['MonthlyCharges'].mean()
total_customers = len(df)
retention = 100 - churn_rate

col1.metric("👥 Customers", total_customers)
col2.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
col3.metric("💰 Avg Charges", f"${avg_charge:.2f}")
col4.metric("📊 Retention", f"{retention:.2f}%")

st.markdown("---")

# ---------------- CHARTS ----------------
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="Contract", color="Churn",
                       title="Churn by Contract Type")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    churn_counts = df['Churn'].value_counts()
    fig2 = px.pie(values=churn_counts.values,
                  names=churn_counts.index,
                  title="Churn Distribution",
                  hole=0.5)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------- USER INPUT ----------------
st.header("🧾 Customer Prediction")

# MAIN INPUTS (important)
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    monthly = st.slider("Monthly Charges", 0.0, 200.0, 70.0)

with col3:
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])

# ADVANCED OPTIONS
with st.expander("⚙️ Advanced Options"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No"])
    support = st.selectbox("Tech Support", ["Yes", "No"])
    payment = st.selectbox("Payment Method",
                           ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])

# ---------------- AUTO FILL ----------------
if st.button("⚡ Use Sample Customer"):
    tenure = 24
    monthly = 85
    contract = "Month-to-month"

# ---------------- PREDICT ----------------
if st.button("🔮 Predict Churn"):

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "Contract": contract,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet,
        "OnlineSecurity": security,
        "TechSupport": support,
        "PaymentMethod": payment
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown("---")

    # ---------------- RESULT ----------------
    st.header("🎯 Prediction Result")

    if prob > 70:
        st.error(f"⚠️ HIGH RISK ({prob:.2f}%)")
        st.write("👉 Suggestion: Offer discount / retention call")

    elif prob > 40:
        st.warning(f"🟡 MEDIUM RISK ({prob:.2f}%)")
        st.write("👉 Suggestion: Engage with offers")

    else:
        st.success(f"✅ LOW RISK ({prob:.2f}%)")
        st.write("👉 Suggestion: No action needed")

    # ---------------- GAUGE ----------------
    gauge = px.pie(values=[prob, 100-prob],
                   names=["Churn", "Safe"],
                   hole=0.7)
    st.plotly_chart(gauge, use_container_width=True)

    # ---------------- REASON ----------------
    st.subheader("🧠 Why this prediction?")
    reasons = []

    if contract == "Month-to-month":
        reasons.append("Month-to-month contract increases churn risk")
    if monthly > 80:
        reasons.append("High monthly charges")
    if tenure < 12:
        reasons.append("Low tenure")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("Stable customer profile")

st.markdown("---")

# ---------------- DATA TABLE ----------------
st.subheader("📋 Sample Customers")
st.dataframe(df.head(10))