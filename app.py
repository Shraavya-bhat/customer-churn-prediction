import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---- LOAD MODEL ----
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# ---- CUSTOM CSS ----
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.block-container {
    padding-top: 1rem;
}

h1, h2, h3 {
    color: white;
}

hr {
    border: 1px solid #222;
}

.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.big-font {
    font-size: 20px;
}

.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["Predict", "About"])

# =========================
# PAGE 1: PREDICTION
# =========================
if page == "Predict":

    st.markdown("<h1 style='text-align:center;'>🚀 Customer Churn Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Advanced ML-powered churn prediction system")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- INPUT SECTIONS ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Customer Info")
        gender = st.radio("Gender", ["Male", "Female"])
        senior = st.radio("Senior Citizen", ["No", "Yes"])
        partner = st.radio("Partner", ["No", "Yes"])
        dependents = st.radio("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure", 0, 72, 12)

    with col2:
        st.markdown("### 📡 Services")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.radio("Online Security", ["No", "Yes"])
        backup = st.radio("Online Backup", ["No", "Yes"])
        device = st.radio("Device Protection", ["No", "Yes"])
        support = st.radio("Tech Support", ["No", "Yes"])
        streaming = st.radio("Streaming Services", ["No", "Yes"])

    st.markdown("<hr>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 💳 Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col4:
        monthly = st.number_input("Monthly Charges", value=70.0)
        total = st.number_input("Total Charges", value=1000.0)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- PREDICT BUTTON ----
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

        st.markdown("## 🎯 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class='result-box' style='background: linear-gradient(90deg, #ff4b4b, #ff0000);'>
            ⚠️ HIGH CHURN RISK<br>{prob:.2%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-box' style='background: linear-gradient(90deg, #00c853, #00e676);'>
            ✅ LOW CHURN RISK<br>{prob:.2%}
            </div>
            """, unsafe_allow_html=True)

        st.progress(float(prob))

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---- EXPLANATION ----
        st.subheader("🧠 Why this prediction?")

        if tenure < 12:
            st.write("- Low tenure increases churn risk")
        if contract == "Month-to-month":
            st.write("- Month-to-month users churn more")
        if monthly > 80:
            st.write("- High monthly charges increase churn")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---- FEATURE IMPORTANCE ----
        st.subheader("📊 Top Factors")

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
            color_continuous_scale=["#ff4b4b", "#ff0000"]
        )

        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 2: ABOUT
# =========================
else:
    st.title("📘 About Project")

    st.write("""
    This project predicts customer churn using Machine Learning.

    🔹 Model: Random Forest  
    🔹 Dataset: Telco Customer Churn  
    🔹 Accuracy: ~80%

    Built using:
    - Streamlit
    - Scikit-learn
    - Plotly
    """)
