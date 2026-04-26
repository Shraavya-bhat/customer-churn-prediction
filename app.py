import pickle
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Load model
model = pickle.load(open("model/churn_model.pkl", "rb"))
cols = pickle.load(open("model/columns.pkl", "rb"))

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# --------------------------
# HEADER
# --------------------------
st.title("🚀 Customer Churn Prediction System")
st.info("This app predicts customer churn using machine learning + business rules.")

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Predict Churn"])

# --------------------------
# DASHBOARD TAB
# --------------------------
with tab1:
    st.subheader("📊 Business Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", "7043")
    col2.metric("Churn Rate", "26.54%")
    col3.metric("Avg Charges", "$64.76")
    col4.metric("Retention", "73.46%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Contract")

        fig = go.Figure(data=[
            go.Bar(name='Stayed', x=['Month-to-month', 'One year', 'Two year'],
                   y=[2200, 1300, 1600]),
            go.Bar(name='Churned', x=['Month-to-month', 'One year', 'Two year'],
                   y=[1700, 200, 100])
        ])
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Distribution")

        fig2 = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[73.5, 26.5],
            hole=.5
        )])
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# PREDICT TAB
# --------------------------
with tab2:
    st.subheader("🧠 Predict Customer Churn")

    # Inputs
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)

    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])

    # Advanced options
    with st.expander("⚙ Advanced Options"):
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No"])
        support = st.selectbox("Tech Support", ["Yes", "No"])

    # Sample button
    if st.button("🔥 Use High Risk Sample"):
        tenure = 2
        monthly = 120.0
        contract = "Month-to-month"
        security = "No"
        support = "No"
        st.success("High-risk sample loaded!")

    # Prediction
    if st.button("🔮 Predict Churn"):

        # Create empty row
        input_data = pd.DataFrame(columns=cols)
        input_data.loc[0] = 0

        # Fill values
        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly

        # Contract encoding
        if contract == "Month-to-month":
            input_data["Contract_Month-to-month"] = 1
        elif contract == "One year":
            input_data["Contract_One year"] = 1
        else:
            input_data["Contract_Two year"] = 1

        # Prediction
        prediction = model.predict_proba(input_data)[0][1]
        percentage = prediction * 100

        # Risk label
        if prediction > 0.6:
            st.error(f"HIGH RISK ({percentage:.2f}%)")
        elif prediction > 0.4:
            st.warning(f"MEDIUM RISK ({percentage:.2f}%)")
        else:
            st.success(f"LOW RISK ({percentage:.2f}%)")

        # Donut chart
        fig = go.Figure(go.Pie(
            values=[percentage, 100 - percentage],
            labels=["Churn Risk", "Safe"],
            hole=0.6
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Explanation
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🧠 Why this result?")
            reasons = []

            if contract == "Month-to-month":
                reasons.append("Flexible contract → higher churn")

            if tenure < 12:
                reasons.append("New customer")

            if monthly > 80:
                reasons.append("High monthly charges")

            if security == "No":
                reasons.append("No security service")

            for r in reasons:
                st.write(f"• {r}")

        with col2:
            st.markdown("### 💡 Suggested Action")

            if percentage >= 50:
                st.info("Offer discount / retention call")
            elif percentage >= 30:
                st.info("Engage with offers")
            else:
                st.info("Customer is stable")

        