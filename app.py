import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ---- CONFIG ----
st.set_page_config(page_title="Churn Predictor", layout="wide")

# ---- LOAD MODEL ----
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---- STYLE ----
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
h1 {text-align: center; font-size: 42px;}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.title("🚀 Customer Churn Prediction System")
st.caption("ML-powered churn prediction using Random Forest • Accuracy ~80%")

st.markdown("---")

# ---- CUSTOMER INFO ----
st.markdown("## 👤 Customer Info")
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    senior = st.radio("Senior Citizen", ["No", "Yes"])
    partner = st.radio("Partner", ["No", "Yes"])

with col2:
    dependents = st.radio("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

st.markdown("---")

# ---- SERVICES ----
st.markdown("## 📡 Services")
col3, col4 = st.columns(2)

with col3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.radio("Online Security", ["No", "Yes"])
    backup = st.radio("Online Backup", ["No", "Yes"])

with col4:
    device = st.radio("Device Protection", ["No", "Yes"])
    support = st.radio("Tech Support", ["No", "Yes"])
    streaming = st.radio("Streaming Services", ["No", "Yes"])

st.markdown("---")

# ---- BILLING ----
st.markdown("## 💳 Billing")
col5, col6 = st.columns(2)

with col5:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col6:
    monthly = st.number_input("Monthly Charges", value=70.0)
    total = st.number_input("Total Charges", value=1000.0)

st.markdown("---")

# ---- PREDICT ----
if st.button("🔍 Predict Churn"):

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
        "DeviceProtection": device,
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

    # ---- RESULT ----
    st.markdown("## 🎯 Prediction Result")
    colA, colB = st.columns([2,1])

    with colA:
        if prediction == 1:
            st.error(f"⚠️ High Churn Risk ({prob:.2%})")
        else:
            st.success(f"✅ Low Churn Risk ({prob:.2%})")

    with colB:
        st.metric("Churn Probability", f"{prob:.2%}")

    st.progress(float(prob))

    st.markdown("---")

    # ---- EXPLANATION ----
    st.subheader("🧠 Why this prediction?")

    if tenure < 12:
        st.write("- Low tenure increases churn risk")
    if contract == "Month-to-month":
        st.write("- Month-to-month users churn more")
    if monthly > 80:
        st.write("- High monthly charges increase churn")

    st.markdown("---")

    # ---- FEATURE IMPORTANCE ----
    st.subheader("📊 Top Factors Affecting Prediction")

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation='h',
        color="Importance",
        color_continuous_scale="reds"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)