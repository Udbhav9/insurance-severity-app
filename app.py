import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Auto Insurance Severity Analysis", page_icon="🚗", layout="wide")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('xgb_model.joblib')
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
    st.sidebar.success("✅ Model Pipeline Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.title("🚗 Auto Insurance Claim Severity Engine")
st.markdown("Predict claim payout amounts at **First Notice of Loss (FNOL)** using pre-incident data.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Policyholder Profile")
    age = st.number_input("Driver Age", min_value=18, max_value=85, value=38)
    months_as_customer = st.number_input("Months as Customer", min_value=0, max_value=600, value=128)
    policy_annual_premium = st.number_input("Policy Annual Premium ($)", min_value=100.0, max_value=5000.0, value=1400.0)
    policy_deductible = st.selectbox("Policy Deductible ($)", [500, 1000, 2000])
    umbrella_limit = st.selectbox("Umbrella Limit ($)", [0, 1000000, 2000000, 3000000])

with col2:
    st.subheader("💥 Incident Metadata")
    incident_severity = st.selectbox("Incident Severity", ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'])
    incident_type = st.selectbox("Incident Type", ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Side Impact', 'Parked Car'])
    collision_type = st.selectbox("Collision Type", ['Front Collision', 'Rear Collision', 'Side Collision', 'MISSING_UNSPECIFIED'])
    number_of_vehicles_involved = st.slider("Vehicles Involved", 1, 6, 2)
    bodily_injuries = st.slider("Bodily Injuries Reported", 0, 5, 1)

with col3:
    st.subheader("📋 Verification & Policy Details")
    policy_state = st.selectbox("Policy State", ['OH', 'IN', 'IL'])
    insured_sex = st.selectbox("Insured Sex", ['FEMALE', 'MALE'])
    insured_education_level = st.selectbox("Education Level", ['JD', 'High School', 'Associate', 'MD', 'Masters', 'PhD', 'College'])
    insured_occupation = st.selectbox("Occupation", ['craft-repair', 'machine-op-inspct', 'prof-specialty', 'tech-support', 'sales', 'exec-managerial', 'protective-serv', 'armed-forces', 'other-service', 'adm-clerical', 'farming-fishing', 'transport-moving', 'priv-house-serv', 'handlers-cleaners'])
    property_damage = st.selectbox("Property Damage Reported", ['NO', 'YES', 'MISSING_UNSPECIFIED'])
    police_report_available = st.selectbox("Police Report Available", ['NO', 'YES', 'MISSING_UNSPECIFIED'])

if st.button("🚀 Calculate Estimated Claim Severity & Risk Tier", use_container_width=True):
    input_data = pd.DataFrame([{
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_deductible': policy_deductible,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'policy_state': policy_state,
        'insured_sex': insured_sex,
        'insured_education_level': insured_education_level,
        'insured_occupation': insured_occupation,
        'incident_type': incident_type,
        'collision_type': collision_type,
        'incident_severity': incident_severity,
        'number_of_vehicles_involved': number_of_vehicles_involved,
        'bodily_injuries': bodily_injuries,
        'property_damage': property_damage,
        'police_report_available': police_report_available,
        'customer_tenure_ratio': months_as_customer / (age * 12 + 1e-5)
    }])

    X_proc = preprocessor.transform(input_data)
    log_pred = model.predict(X_proc)[0]
    predicted_dollar = np.expm1(log_pred)

    st.divider()
    st.subheader("📊 Output & Operational Recommendation")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Estimated Claim Payout ($)", f"${predicted_dollar:,.2f}")

    with res_col2:
        if predicted_dollar < 45000:
            st.success("🟢 **TIER 1: LOW RISK (<$45,000)**")
            st.info("⚡ **Operational Path:** Fast-track to **Straight-Through Processing (STP)** for automated claim approval.")
        elif 45000 <= predicted_dollar <= 62000:
            st.warning("🟡 **TIER 2: MEDIUM RISK ($45,000 - $62,000)**")
            st.info("🔍 **Operational Path:** Assign to **Standard Adjuster Queue** for damage verification.")
        else:
            st.error("🔴 **TIER 3: HIGH RISK (>$62,000)**")
            st.info("🚨 **Operational Path:** Escalate immediately to **Senior Adjuster & SIU Fraud Audit**.")
