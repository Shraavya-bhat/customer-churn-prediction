import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import time

# ---- CONFIG ----
st.set_page_config(page_title="Churn Risk Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model/churn_model.pkl", "rb"))
        columns = pickle.load(open("model/columns.pkl", "rb"))
        return model, columns
    except:
        st.error("Model files not found. Please ensure model/churn_model.pkl and model/columns.pkl exist.")
        st.stop()

model, columns = load_model()

# ---- CLEAN CSS ----
st.markdown("""
<style>
/* Remove ALL default padding/margin noise */
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* Hide Streamlit default elements that add clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Clean sections */
.section {
    background: #1a1f2e;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    border: 1px solid #2d3748;
}

/* Inputs */
div[data-baseweb="select"] {
    margin-bottom: 12px;
}

/* Buttons */
.stButton button {
    background: #3b82f6;
    border-radius: 8px;
    height: 44px;
    font-size: 15px;
    font-weight: 500;
    border: none;
    width: 100%;
}
.stButton button:hover {
    background: #2563eb;
}

/* Sample button */
.sample-btn button {
    background: #6366f1 !important;
}

/* Result card */
.result-card {
    background: #1a1f2e;
    padding: 24px;
    border-radius: 12px;
    border: 2px solid #3b82f6;
    margin: 16px 0;
}

/* Metric boxes */
.metric-box {
    text-align: center;
    padding: 16px;
    background: #0f172a;
    border-radius: 10px;
}

/* Factor bars */
.factor-row {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    background: #1a1f2e;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 3px solid;
}

/* Recommendation cards */
.rec-card {
    background: #1a1f2e;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    border: 1px solid #2d3748;
}

/* What-if */
.sim-box {
    background: #1a1f2e;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #2d3748;
    margin: 20px 0;
}

/* Remove expander border/padding noise */
.streamlit-expanderHeader {
    background: #1a1f2e !important;
    border-radius: 8px !important;
    border: 1px solid #2d3748 !important;
    padding: 12px 16px !important;
}
.streamlit-expanderContent {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.title("Churn Risk Dashboard")
st.caption("Predict how likely a customer is to leave — and what to do about it")

# ---- SAMPLE BUTTON ----
if st.button("🧪 Try Demo (High-Risk Customer)", type="secondary", use_container_width=True):
    st.session_state.sample = True
    st.session_state.gender = "Male"
    st.session_state.senior = "No"
    st.session_state.partner = "No"
    st.session_state.dependents = "No"
    st.session_state.tenure = 8
    st.session_state.internet = "Fiber optic"
    st.session_state.security = "No"
    st.session_state.backup = "No"
    st.session_state.protection = "No"
    st.session_state.support = "No"
    st.session_state.streaming = "Yes"
    st.session_state.contract = "Month-to-month"
    st.session_state.payment = "Electronic check"
    st.session_state.monthly = 95.0
    st.session_state.total = 760.0
    st.rerun()

st.markdown("---")

# =========================
# CUSTOMER INFO
# =========================

st.markdown("**Customer Info**")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], 
                         index=0, key="gender" if "sample" in st.session_state else None)
    partner = st.selectbox("Has partner", ["No", "Yes"], 
                          index=0, key="partner" if "sample" in st.session_state else None)
with col2:
    senior = st.selectbox("Senior citizen", ["No", "Yes"], 
                         index=0, key="senior" if "sample" in st.session_state else None)
    dependents = st.selectbox("Has dependents", ["No", "Yes"], 
                               index=0, key="dependents" if "sample" in st.session_state else None)
with col3:
    tenure = st.slider("Tenure (months)", 0, 72, 
                      st.session_state.get("tenure", 12) if "sample" in st.session_state else 12,
                      key="tenure_slider")

st.markdown("---")

# =========================
# ADVANCED SETTINGS (Collapsed)
# =========================

with st.expander("More options"):
    c1, c2 = st.columns(2)
    with c1:
        internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"], 
                               index=0, key="internet" if "sample" in st.session_state else None)
        security = st.selectbox("Online security", ["No", "Yes"], 
                               index=0, key="security" if "sample" in st.session_state else None)
        backup = st.selectbox("Online backup", ["No", "Yes"], 
                             index=0, key="backup" if "sample" in st.session_state else None)
    with c2:
        protection = st.selectbox("Device protection", ["No", "Yes"], 
                                 index=0, key="protection" if "sample" in st.session_state else None)
        support = st.selectbox("Tech support", ["No", "Yes"], 
                              index=0, key="support" if "sample" in st.session_state else None)
        streaming = st.selectbox("Streaming TV/Movies", ["No", "Yes"], 
                                  index=0, key="streaming" if "sample" in st.session_state else None)
    
    c3, c4 = st.columns(2)
    with c3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], 
                               index=0, key="contract" if "sample" in st.session_state else None)
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], index=0, key="payment" if "sample" in st.session_state else None)
    with c4:
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, 
                                   value=st.session_state.get("monthly", 70.0) if "sample" in st.session_state else 70.0,
                                   step=5.0, key="monthly_input")
        total = st.number_input("Total charges ($)", min_value=0.0, 
                               value=st.session_state.get("total", 1000.0) if "sample" in st.session_state else 1000.0,
                               step=50.0, key="total_input")

# Default values if no sample loaded
if "sample" not in st.session_state:
    internet = st.session_state.get("internet", "DSL")
    security = st.session_state.get("security", "No")
    backup = st.session_state.get("backup", "No")
    protection = st.session_state.get("protection", "No")
    support = st.session_state.get("support", "No")
    streaming = st.session_state.get("streaming", "No")
    contract = st.session_state.get("contract", "Month-to-month")
    payment = st.session_state.get("payment", "Electronic check")
    monthly = st.session_state.get("monthly", 70.0)
    total = st.session_state.get("total", 1000.0)

# =========================
# VALIDATION
# =========================

if monthly <= 0:
    st.error("Monthly charges must be greater than $0")
    st.stop()
if total < monthly:
    st.error("Total charges cannot be less than one month's charge")
    st.stop()

# =========================
# PREDICT
# =========================

POPULATION_AVG = 27  # 27%

def get_risk(prob):
    if prob < 25:
        return "Low", "#22c55e", "Customer is stable", "✅"
    elif prob < 50:
        return "Medium", "#f59e0b", "Customer may leave — reach out soon", "⚠️"
    elif prob < 75:
        return "High", "#ef4444", "Customer is likely to leave — act now", "🔴"
    else:
        return "Critical", "#dc2626", "Customer will probably leave within 30 days", "🚨"

def get_factors(inputs):
    factors = []
    
    if inputs["Contract"] == "Month-to-month":
        factors.append(("No long-term contract", 18, "bad", "Customers without contracts leave more often"))
    elif inputs["Contract"] == "One year":
        factors.append(("One-year contract", -12, "good", "Contracts reduce churn"))
    else:
        factors.append(("Two-year contract", -22, "good", "Long contracts strongly reduce churn"))
    
    if inputs["PaymentMethod"] == "Electronic check":
        factors.append(("Pays by electronic check", 12, "bad", "Manual payments correlate with higher churn"))
    elif "automatic" in inputs["PaymentMethod"]:
        factors.append(("Auto-pay enabled", -8, "good", "Auto-pay makes leaving harder"))
    
    if inputs["InternetService"] == "Fiber optic":
        factors.append(("Uses fiber optic", 10, "bad", "Premium service = higher price sensitivity"))
    elif inputs["InternetService"] == "DSL":
        factors.append(("Uses DSL", -5, "good", "Lower cost = less reason to switch"))
    
    if inputs["tenure"] < 12:
        factors.append(("New customer (< 1 year)", 15, "bad", "New customers haven't built loyalty yet"))
    elif inputs["tenure"] > 36:
        factors.append(("Long-time customer", -10, "good", "Loyal customers stay longer"))
    
    if inputs["OnlineSecurity"] == "No":
        factors.append(("No security service", 8, "bad", "Missing services = less invested"))
    if inputs["TechSupport"] == "No":
        factors.append(("No tech support", 7, "bad", "Frustration without support drives churn"))
    
    if inputs["MonthlyCharges"] > 85:
        factors.append((f"High bill (${inputs['MonthlyCharges']:.0f})", 9, "bad", "Expensive plans make customers price-shop"))
    
    factors.sort(key=lambda x: abs(x[1]), reverse=True)
    return factors[:5]

def get_actions(risk, inputs):
    actions = []
    
    if inputs["Contract"] == "Month-to-month":
        actions.append({
            "what": "Offer a 1-year contract with 10% discount",
            "why": "Locks customer in, reduces chance of leaving",
            "cost": f"${inputs['MonthlyCharges'] * 12 * 0.10:.0f}/year",
            "urgency": "high" if risk in ["High", "Critical"] else "medium"
        })
    
    if inputs["PaymentMethod"] == "Electronic check":
        actions.append({
            "what": "Switch to auto-pay + $5/month credit",
            "why": "Auto-pay removes the 'cancel' friction",
            "cost": "$60/year credit",
            "urgency": "high" if risk in ["High", "Critical"] else "medium"
        })
    
    if inputs["TechSupport"] == "No":
        actions.append({
            "what": "Add tech support for $5/month",
            "why": "Support reduces frustration-driven cancellations",
            "cost": "$60/year",
            "urgency": "medium"
        })
    
    if inputs["OnlineSecurity"] == "No":
        actions.append({
            "what": "Free online security for 3 months",
            "why": "More services = more reasons to stay",
            "cost": "$30",
            "urgency": "medium"
        })
    
    if inputs["MonthlyCharges"] > 85:
        actions.append({
            "what": "Offer loyalty discount or plan review",
            "why": "Price is a top reason people switch",
            "cost": "Flexible",
            "urgency": "high"
        })
    
    order = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda x: order.get(x["urgency"], 3))
    return actions[:3]

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Analyze Churn Risk", use_container_width=True):
    
    with st.spinner("Analyzing..."):
        time.sleep(0.6)
    
    inputs = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": streaming,
        "StreamingMovies": streaming, "Contract": contract,
        "PaperlessBilling": "Yes", "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total
    }
    
    df = pd.DataFrame([inputs])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    
    prob = model.predict_proba(df)[0][1] * 100
    risk, color, desc, icon = get_risk(prob)
    factors = get_factors(inputs)
    actions = get_actions(risk, inputs)
    
    diff = prob - POPULATION_AVG
    diff_word = "higher" if diff > 0 else "lower"
    
    ltv = monthly * min(max(tenure, 12), 24)
    
    # =========================
    # RESULT CARD
    # =========================
    
    st.markdown("---")
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 120px; text-align: center; padding: 16px; background: #0f172a; border-radius: 10px;">
                <div style="font-size: 28px; color: {color};">{icon}</div>
                <div style="font-size: 20px; font-weight: 700; color: {color}; margin-top: 4px;">{risk} Risk</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">{desc}</div>
            </div>
            <div style="flex: 1; min-width: 120px; text-align: center; padding: 16px; background: #0f172a; border-radius: 10px;">
                <div style="font-size: 32px; font-weight: 700; color: {color};">{prob:.0f}%</div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">Chance of leaving</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">{abs(diff):.0f} points {diff_word} than average (27%)</div>
            </div>
            <div style="flex: 1; min-width: 120px; text-align: center; padding: 16px; background: #0f172a; border-radius: 10px;">
                <div style="font-size: 28px; font-weight: 700; color: #3b82f6;">${ltv:.0f}</div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">Money at risk</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">If customer leaves</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        title={'text': "Churn Probability", 'font': {'size': 13, 'color': '#94a3b8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': "#0f172a",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 25], 'color': "rgba(34,197,94,0.1)"},
                {'range': [25, 50], 'color': "rgba(245,158,11,0.1)"},
                {'range': [50, 75], 'color': "rgba(239,68,68,0.1)"},
                {'range': [75, 100], 'color': "rgba(220,38,38,0.1)"}
            ],
            'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': prob}
        }
    ))
    fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # =========================
    # WHY
    # =========================
    
    st.markdown("**Why this score?**")
    st.caption("Top reasons this customer might leave")
    
    for name, points, kind, explanation in factors:
        bar_color = "#ef4444" if kind == "bad" else "#22c55e"
        width = min(abs(points) * 2.5, 100)
        
        st.markdown(f"""
        <div class="factor-row" style="border-left-color: {bar_color};">
            <div style="width: 32%; font-size: 13px; color: #e2e8f0;">{name}</div>
            <div style="width: 35%; padding-right: 12px;">
                <div style="background: #334155; height: 6px; border-radius: 3px;">
                    <div style="width: {width}%; height: 100%; background: {bar_color}; border-radius: 3px;"></div>
                </div>
            </div>
            <div style="width: 33%; font-size: 12px; color: #94a3b8; text-align: right;">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================
    # WHAT TO DO
    # =========================
    
    st.markdown("---")
    st.markdown("**What to do**")
    st.caption("Best actions to keep this customer")
    
    for i, action in enumerate(actions, 1):
        badge_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}[action["urgency"]]
        badge_text = {"high": "URGENT", "medium": "RECOMMENDED", "low": "OPTIONAL"}[action["urgency"]]
        
        st.markdown(f"""
        <div class="rec-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="font-size: 14px; font-weight: 600; color: #f1f5f9;">{i}. {action["what"]}</div>
                <div style="background: {badge_color}20; color: {badge_color}; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">{badge_text}</div>
            </div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">{action["why"]}</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">💰 Cost: {action["cost"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================
    # WHAT-IF
    # =========================
    
    st.markdown("---")
    st.markdown("**Test changes**")
    st.caption("See how small changes affect churn risk")
    
    st.markdown(f"""
    <div class="sim-box">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">Current: {contract} · Tech support: {support} · {payment} · ${monthly:.0f}/mo</div>
    </div>
    """, unsafe_allow_html=True)
    
    s1, s2, s3 = st.columns(3)
    with s1:
        sim_contract = st.selectbox("Contract", ["Keep same", "One year", "Two year"], key="sim_c")
    with s2:
        sim_support = st.selectbox("Tech support", ["Keep same", "Add it"], key="sim_s")
    with s3:
        sim_payment = st.selectbox("Payment", ["Keep same", "Bank transfer", "Credit card"], key="sim_p")
    
    if st.button("See new risk", key="sim_btn"):
        with st.spinner("Calculating..."):
            time.sleep(0.4)
        
        sim_inputs = inputs.copy()
        if sim_contract != "Keep same":
            sim_inputs["Contract"] = sim_contract
        if sim_support == "Add it":
            sim_inputs["TechSupport"] = "Yes"
        if sim_payment != "Keep same":
            sim_inputs["PaymentMethod"] = sim_payment + " (automatic)" if sim_payment != "Keep same" else sim_inputs["PaymentMethod"]
        
        sim_df = pd.DataFrame([sim_inputs])
        sim_df = pd.get_dummies(sim_df)
        sim_df = sim_df.reindex(columns=columns, fill_value=0)
        
        sim_prob = model.predict_proba(sim_df)[0][1] * 100
        sim_risk, sim_color, _, _ = get_risk(sim_prob)
        change = sim_prob - prob
        
        change_color = "#22c55e" if change < 0 else "#ef4444"
        change_icon = "↓" if change < 0 else "↑"
        
        st.markdown(f"""
        <div style="display: flex; gap: 16px; align-items: center; justify-content: center; margin-top: 16px;">
            <div style="text-align: center; padding: 16px 24px; background: #334155; border-radius: 10px;">
                <div style="font-size: 12px; color: #94a3b8;">Before</div>
                <div style="font-size: 24px; font-weight: 700; color: {color};">{prob:.0f}%</div>
                <div style="font-size: 12px; color: {color};">{risk}</div>
            </div>
            <div style="font-size: 20px; color: #64748b;">→</div>
            <div style="text-align: center; padding: 16px 24px; background: #1e3a5f; border-radius: 10px; border: 2px solid #3b82f6;">
                <div style="font-size: 12px; color: #94a3b8;">After</div>
                <div style="font-size: 24px; font-weight: 700; color: {sim_color};">{sim_prob:.0f}%</div>
                <div style="font-size: 12px; color: {sim_color};">{sim_risk}</div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 12px; color: {change_color}; font-size: 14px; font-weight: 600;">
            {change_icon} {abs(change):.0f} percentage points
        </div>
        """, unsafe_allow_html=True)