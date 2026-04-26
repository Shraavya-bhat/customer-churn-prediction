import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# --------------------------
# TRAIN / LOAD MODEL
# --------------------------
@st.cache_resource
def load_or_train_model():
    """Train a model on synthetic data matching Telco Churn schema"""
    np.random.seed(42)
    n = 5000
    
    # Generate realistic synthetic training data
    df = pd.DataFrame({
        'tenure': np.random.randint(0, 73, n),
        'MonthlyCharges': np.random.uniform(18, 118, n),
        'TotalCharges': np.random.uniform(18, 8684, n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n, p=[0.28, 0.72]),
        'TechSupport': np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No'], n, p=[0.34, 0.66]),
        'DeviceProtection': np.random.choice(['Yes', 'No'], n, p=[0.34, 0.66]),
        'StreamingTV': np.random.choice(['Yes', 'No'], n, p=[0.38, 0.62]),
        'StreamingMovies': np.random.choice(['Yes', 'No'], n, p=[0.39, 0.61]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
    })
    
    # Realistic churn logic (higher churn for month-to-month, high charges, no security, low tenure)
    churn_prob = (
        0.10 +
        (df['Contract'] == 'Month-to-month') * 0.30 +
        (df['tenure'] < 12) * 0.20 +
        (df['MonthlyCharges'] > 80) * 0.15 +
        (df['OnlineSecurity'] == 'No') * 0.10 +
        (df['TechSupport'] == 'No') * 0.08 +
        (df['InternetService'] == 'Fiber optic') * 0.10 +
        (df['PaymentMethod'] == 'Electronic check') * 0.10
    )
    churn_prob = np.clip(churn_prob, 0, 0.95)
    df['Churn'] = (np.random.random(n) < churn_prob).astype(int)
    
    # Preprocess
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode categoricals
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    return model, encoders, df

model, encoders, train_df = load_or_train_model()

# --------------------------
# SESSION STATE FOR SAMPLE BUTTON
# --------------------------
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None

# --------------------------
# HEADER
# --------------------------
st.title("🚀 Customer Churn Prediction System")
st.info("Real ML model (Random Forest) trained on 5,000 synthetic customer records.")

tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Predict Churn"])

# --------------------------
# DASHBOARD TAB
# --------------------------
with tab1:
    st.subheader("📊 Business Overview")
    
    total = len(train_df)
    churn_rate = train_df['Churn'].mean() * 100
    retention = 100 - churn_rate
    avg_charges = train_df['MonthlyCharges'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{total:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg Charges", f"${avg_charges:.2f}")
    col4.metric("Retention", f"{retention:.1f}%")
    
    st.markdown("---")
    
    # Churn by Contract
    contract_churn = train_df.groupby('Contract')['Churn'].agg(['sum', 'count'])
    contract_churn['stayed'] = contract_churn['count'] - contract_churn['sum']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn by Contract")
        fig = go.Figure(data=[
            go.Bar(name='Stayed', x=contract_churn.index, y=contract_churn['stayed'], marker_color='#2ecc71'),
            go.Bar(name='Churned', x=contract_churn.index, y=contract_churn['sum'], marker_color='#e74c3c')
        ])
        fig.update_layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn Distribution")
        fig2 = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[retention, churn_rate],
            hole=.5,
            marker_colors=['#2ecc71', '#e74c3c']
        )])
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🔍 What Drives Churn? (Feature Importance)")
    
    importance_df = pd.DataFrame({
        'Feature': train_df.drop('Churn', axis=1).columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig3 = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#3498db'
    ))
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------
# PREDICT TAB
# --------------------------
with tab2:
    st.subheader("🧠 Predict Customer Churn")
    
    # Default values (either from sample or defaults)
    defaults = {
        'tenure': 2 if st.session_state.sample_loaded else 12,
        'monthly': 120.0 if st.session_state.sample_loaded else 70.0,
        'contract': "Month-to-month" if st.session_state.sample_loaded else "Month-to-month",
        'internet': "Fiber optic" if st.session_state.sample_loaded else "DSL",
        'security': "No" if st.session_state.sample_loaded else "No",
        'support': "No" if st.session_state.sample_loaded else "No",
        'payment': "Electronic check" if st.session_state.sample_loaded else "Electronic check",
        'gender': "Male" if st.session_state.sample_loaded else "Male",
        'senior': 1 if st.session_state.sample_loaded else 0,
        'partner': "No" if st.session_state.sample_loaded else "No",
        'dependents': "No" if st.session_state.sample_loaded else "No",
        'phone': "Yes" if st.session_state.sample_loaded else "Yes",
        'multiple': "No" if st.session_state.sample_loaded else "No",
        'backup': "No" if st.session_state.sample_loaded else "No",
        'protection': "No" if st.session_state.sample_loaded else "No",
        'tv': "No" if st.session_state.sample_loaded else "No",
        'movies': "No" if st.session_state.sample_loaded else "No",
        'paperless': "Yes" if st.session_state.sample_loaded else "Yes",
    }
    
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, defaults['tenure'], key='tenure')
    with col2:
        monthly = st.slider("Monthly Charges ($)", 0.0, 150.0, defaults['monthly'], key='monthly')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], 
                               index=["Month-to-month", "One year", "Two year"].index(defaults['contract']))
    with col2:
        payment = st.selectbox("Payment Method", 
                              ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                              index=0 if defaults['payment'] == "Electronic check" else 1)
    with col3:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                               index=["DSL", "Fiber optic", "No"].index(defaults['internet']))
    
    with st.expander("⚙ Advanced Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            security = st.selectbox("Online Security", ["Yes", "No"], index=["Yes", "No"].index(defaults['security']))
            support = st.selectbox("Tech Support", ["Yes", "No"], index=["Yes", "No"].index(defaults['support']))
            backup = st.selectbox("Online Backup", ["Yes", "No"], index=["Yes", "No"].index(defaults['backup']))
        with col2:
            protection = st.selectbox("Device Protection", ["Yes", "No"], index=["Yes", "No"].index(defaults['protection']))
            tv = st.selectbox("Streaming TV", ["Yes", "No"], index=["Yes", "No"].index(defaults['tv']))
            movies = st.selectbox("Streaming Movies", ["Yes", "No"], index=["Yes", "No"].index(defaults['movies']))
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(defaults['gender']))
            senior = st.selectbox("Senior Citizen", [0, 1], index=[0, 1].index(defaults['senior']))
            partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(defaults['partner']))
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(defaults['dependents']))
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(defaults['phone']))
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], 
                                   index=["Yes", "No", "No phone service"].index(defaults['multiple']))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(defaults['paperless']))
    
    # --------------------------
    # SAMPLE BUTTON (FIXED WITH SESSION STATE)
    # --------------------------
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🔥 Use High Risk Sample"):
            st.session_state.sample_loaded = True
            st.rerun()
    with col_btn2:
        if st.button("🔄 Reset to Defaults"):
            st.session_state.sample_loaded = False
            st.rerun()
    
    # --------------------------
    # PREDICTION LOGIC (REAL ML)
    # --------------------------
    if st.button("🔮 Predict Churn", type="primary"):
        
        # Build input dataframe matching training schema
        input_data = pd.DataFrame([{
            'tenure': tenure,
            'MonthlyCharges': monthly,
            'TotalCharges': monthly * tenure,  # Approximation
            'Contract': contract,
            'InternetService': internet,
            'OnlineSecurity': security,
            'TechSupport': support,
            'PaymentMethod': payment,
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'PaperlessBilling': paperless,
        }])
        
        # Encode using same encoders
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in encoders:
                # Handle unseen categories
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
                )
                input_data[col] = encoders[col].transform(input_data[col])
        
        # Predict
        prob = model.predict_proba(input_data)[0][1]
        percentage = int(prob * 100)
        prediction = model.predict(input_data)[0]
        
        st.session_state.pred_result = {
            'percentage': percentage,
            'prob': prob,
            'input_data': input_data,
            'contract': contract,
            'tenure': tenure,
            'monthly': monthly,
            'security': security,
            'support': support,
            'internet': internet
        }
    
    # --------------------------
    # DISPLAY RESULT
    # --------------------------
    if st.session_state.pred_result:
        res = st.session_state.pred_result
        percentage = res['percentage']
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        # Risk badge
        if percentage >= 60:
            st.error(f"⚠️ HIGH RISK OF CHURN ({percentage}%)")
            color = "#e74c3c"
        elif percentage >= 30:
            st.warning(f"🟡 MEDIUM RISK OF CHURN ({percentage}%)")
            color = "#f39c12"
        else:
            st.success(f"✅ LOW RISK OF CHURN ({percentage}%)")
            color = "#2ecc71"
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': '#d5f5e3'},
                    {'range': [30, 60], 'color': '#fdebd0'},
                    {'range': [60, 100], 'color': '#fadbd8'}
                ],
                'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': percentage}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Explanation + Action
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🧠 Why this result?")
            reasons = []
            if res['contract'] == "Month-to-month":
                reasons.append("📋 **Flexible contract** → Month-to-month customers churn 3x more")
            if res['tenure'] < 12:
                reasons.append("🆕 **New customer** → First-year churn is highest")
            if res['monthly'] > 80:
                reasons.append("💰 **High monthly charges** → Price-sensitive customers leave more")
            if res['security'] == "No":
                reasons.append("🔒 **No online security** → Less invested in the service")
            if res['support'] == "No":
                reasons.append("🛠️ **No tech support** → Higher frustration when issues occur")
            if res['internet'] == "Fiber optic":
                reasons.append("🌐 **Fiber optic** → Often paired with higher prices and expectations")
            
            if not reasons:
                reasons.append("✅ Customer profile shows strong retention signals")
            
            for r in reasons:
                st.write(r)
        
        with col2:
            st.markdown("### 💡 Suggested Action")
            if percentage >= 60:
                st.info("🚨 **Immediate retention call**\n- Offer 20% discount\n- Assign dedicated support rep\n- Waive next month fee")
            elif percentage >= 30:
                st.info("⚡ **Proactive engagement**\n- Send loyalty rewards\n- Offer contract upgrade incentive\n- Check-in survey")
            else:
                st.info("✅ **Customer is stable**\n- Maintain service quality\n- Upsell additional services\n- Request referral")
        
        # SHAP-style feature contribution (simplified)
        st.markdown("---")
        st.markdown("### 📊 Key Risk Factors")
        
        risk_factors = []
        if res['contract'] == "Month-to-month": risk_factors.append(("Contract: Month-to-month", 25))
        if res['tenure'] < 12: risk_factors.append(("Tenure < 12 months", 20))
        if res['monthly'] > 80: risk_factors.append(("High monthly charges", 15))
        if res['security'] == "No": risk_factors.append(("No security service", 10))
        if res['support'] == "No": risk_factors.append(("No tech support", 10))
        if res['internet'] == "Fiber optic": risk_factors.append(("Fiber optic service", 8))
        
        if risk_factors:
            factors_df = pd.DataFrame(risk_factors, columns=['Factor', 'Impact'])
            fig_bar = go.Figure(go.Bar(
                x=factors_df['Impact'],
                y=factors_df['Factor'],
                orientation='h',
                marker_color='#e74c3c'
            ))
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=250)
            st.plotly_chart(fig_bar, use_container_width=True)