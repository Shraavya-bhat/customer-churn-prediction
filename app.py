import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/Telco-Customer-Churn.csv")

df = load_data()

# ---------------- HEADER ----------------
st.title("🚀 Customer Churn Prediction System")
st.caption("This app predicts whether a customer will leave (churn) based on their service usage and billing behavior.")

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Predict Churn"])

# =========================================================
# 📊 DASHBOARD TAB
# =========================================================
with tab1:

    st.subheader("📊 Business Overview")

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

    with st.expander("📋 View Sample Customer Data"):
        st.dataframe(df.head(10))


# =========================================================
# 🔮 PREDICTION TAB
# =========================================================
with tab2:

    st.subheader("🔮 Predict Customer Churn")

    # -------- STEP 1 --------
    st.markdown("### Step 1: Basic Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        monthly = st.slider("Monthly Charges", 0.0, 200.0, 70.0)

    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])

    # -------- STEP 2 --------
    st.markdown("### Step 2: Additional Details")

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

    # -------- BUTTONS --------
    col1, col2 = st.columns([1, 2])

    with col1:
        sample = st.button("⚡ Use Sample")

    with col2:
        predict_btn = st.button("🔮 Predict Churn")

    if sample:
        tenure = 24
        monthly = 90
        contract = "Month-to-month"

    # -------- PREDICTION --------
    if predict_btn:

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
        st.subheader("🎯 Prediction Result")

        # -------- RESULT TEXT --------
        if prob > 70:
            st.error(f"⚠️ HIGH RISK ({prob:.2f}%)")
            st.write("👉 Customer likely to churn. Offer retention strategies like discounts.")

        elif prob > 40:
            st.warning(f"🟡 MEDIUM RISK ({prob:.2f}%)")
            st.write("👉 Customer may churn. Engage with offers or support.")

        else:
            st.success(f"✅ LOW RISK ({prob:.2f}%)")
            st.write("👉 Customer likely to stay. No action needed.")

        # -------- GAUGE --------
        fig = px.pie(values=[prob, 100-prob],
                     names=["Churn", "Safe"],
                     hole=0.7)
        st.plotly_chart(fig, use_container_width=True)

        # -------- EXPLANATION --------
        st.subheader("🧠 Why this prediction?")

        reasons = []

        if contract == "Month-to-month":
            reasons.append("Customers with month-to-month contracts churn more frequently.")

        if monthly > 80:
            reasons.append("Higher monthly charges increase churn risk.")

        if tenure < 12:
            reasons.append("New customers are more likely to leave early.")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("Customer profile appears stable.")

        # -------- WHAT IS HAPPENING --------
        st.markdown("---")
        st.subheader("📘 What is happening in this app?")

        st.write("""
This system uses a Machine Learning model trained on telecom customer data.

🔹 It analyzes:
- Customer tenure (how long they stayed)
- Monthly spending
- Contract type
- Services used (internet, support, etc.)

🔹 Based on patterns from past customers, the model:
- Learns who left (churned)
- Learns who stayed

🔹 Then it predicts:
👉 Will this customer leave or stay?

🔹 The probability (e.g., 42%) shows how risky the customer is.

🔹 Businesses use this to:
- Reduce customer loss
- Improve retention strategies
- Increase revenue
""")