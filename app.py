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
st.caption("Predict if a customer will leave based on behavior & usage")

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Predict Churn"])

# =========================================================
# 📊 DASHBOARD
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
                           title="Churn by Contract")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_counts = df['Churn'].value_counts()
        fig2 = px.pie(values=churn_counts.values,
                      names=churn_counts.index,
                      hole=0.5,
                      title="Churn Distribution")
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 🔮 PREDICTION
# =========================================================
with tab2:

    st.subheader("🔮 Predict Customer Churn")

    # -------- BASIC INPUT --------
    st.markdown("### Basic Info")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        monthly = st.slider("Monthly Charges", 0.0, 200.0, 70.0)

    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])

    # -------- ADVANCED --------
    with st.expander("⚙️ Advanced Options"):

        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        internet = st.selectbox("Internet Service",
                                ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No"])
        support = st.selectbox("Tech Support", ["Yes", "No"])
        payment = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)",
                                "Credit card (automatic)"])

    st.markdown("")

    col1, col2 = st.columns([1, 2])

    with col1:
        sample = st.button("⚡ Use High Risk Sample")

    with col2:
        predict_btn = st.button("🔮 Predict Churn")

    # -------- HIGH RISK SAMPLE --------
    if sample:
        tenure = 1
        monthly = 120
        contract = "Month-to-month"
        internet = "Fiber optic"
        security = "No"
        support = "No"

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

        # -------- MODEL OUTPUT --------
        prob = model.predict_proba(input_df)[0][1] * 100

        # -------- BOOST (FIX MODEL WEAKNESS) --------
        if contract == "Month-to-month":
            prob += 15
        if monthly > 80:
            prob += 10
        if tenure < 12:
            prob += 10

        prob = min(prob, 100)

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        # -------- RESULT --------
        if prob > 50:
            st.error(f"⚠️ HIGH RISK ({prob:.2f}%)")

        elif prob > 30:
            st.warning(f"🟡 MEDIUM RISK ({prob:.2f}%)")

        else:
            st.success(f"✅ LOW RISK ({prob:.2f}%)")

        # -------- VISUAL --------
        fig = px.pie(values=[prob, 100-prob],
                     names=["Churn Risk", "Safe"],
                     hole=0.7)
        st.plotly_chart(fig, use_container_width=True)

        # -------- WHY --------
        st.subheader("🧠 Why this result?")

        if contract == "Month-to-month":
            st.write("• Flexible contract → higher churn")

        if monthly > 80:
            st.write("• High monthly charges")

        if tenure < 12:
            st.write("• New customer")

        # -------- ACTION --------
        st.subheader("💡 Suggested Action")

        if prob > 50:
            st.info("Offer discount or call customer")

        elif prob > 30:
            st.info("Engage with offers")

        else:
            st.info("Customer is stable")

        # -------- SIMPLE EXPLANATION --------
        st.subheader("📘 How it works")

        st.write("""
- Model trained on past telecom customer data  
- Learns patterns of churn vs retention  
- Predicts probability of churn  
""")