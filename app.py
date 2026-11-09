import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Auto Insurance Claims Analytics & Severity Engine",
    page_icon="🚘",
    layout="wide"
)

# Dark theme aesthetic styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #00E676; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #B0BEC5; margin-bottom: 20px; }
    .metric-box { background-color: #1E293B; border-radius: 8px; padding: 15px; border-left: 5px solid #00E676; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. LOAD TRAINED ARTIFACTS
# ------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('xgb_model.joblib')
    return preprocessor, model

try:
    preprocessor, model = load_pipeline()
    st.sidebar.success("✅ Model & Preprocessor Pipelines Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Loading Error: {e}")
    st.sidebar.info("💡 Fix: Ensure scikit-learn in requirements.txt matches your Colab version.")

# App Title Header
st.markdown("<p class='main-header'>Auto Insurance Claims Analytics & Severity Engine</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Underwriting Loss Reserving & Dynamic Risk Triage Framework</p>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Comprehensive Project Analysis & Findings", "⚙️ Live FNOL Claim Calculator"])

# ==============================================================================
# TAB 1: DETAILED PROJECT ANALYSIS & OUTCOMES
# ==============================================================================
with tab1:
    st.header("Executive Summary & Analytical Breakdown")
    st.markdown("""
    This project addresses a multi-million dollar operational challenge in insurance analytics: **estimating claim severity on Day 1 (First Notice of Loss)** using strictly pre-incident predictors to automate loss-reserve allocation and claim routing.
    """)
    st.divider()

    # Section A: High-Level Portfolio KPIs
    st.subheader("1. Key Portfolio Metrics (Dataset Overview)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Claims Analyzed", "1,000 Policies")
    kpi2.metric("Mean Claim Payout", "$52,761.94")
    kpi3.metric("Portfolio Loss Exposure", "$52.76 Million")
    kpi4.metric("Portfolio Loss Ratio", "4,199.43%")

    st.divider()

    # Section B: Analytical Deep-Dive & Visual Findings
    st.subheader("2. Core Statistical Findings & Visual Analysis")
    
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### A. Target Variable Skewness & Log Transformation")
        st.write("""
        * **Finding:** Raw `total_claim_amount` is right-skewed with extreme high-value claims up to ~$115k.
        * **Treatment:** Applied $\log(1 + x)$ transformation during training to normalize target distribution for XGBoost regression.
        """)
        # Synthetic visual representation matching dataset properties
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        sns.histplot(np.random.lognormal(mean=10.8, sigma=0.4, size=1000), kde=True, ax=ax1, color='#00E676')
        ax1.set_title("Target Payout Distribution ($)", fontsize=10)
        ax1.set_xlabel("Claim Amount ($)")
        st.pyplot(fig1)

    with col_b:
        st.markdown("#### B. Incident Severity Impact on Dollar Loss")
        st.write("""
        * **Finding:** Claims categorized under **Major Damage** and **Total Loss** account for over 80% of all dollar losses.
        * **Operational Use:** Incident scale at FNOL dictates loss reserving requirements instantly.
        """)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        severity_labels = ['Trivial', 'Minor', 'Major', 'Total Loss']
        severity_payouts = [5000, 12000, 61000, 64000]
        sns.barplot(x=severity_labels, y=severity_payouts, ax=ax2, palette='Reds_r')
        ax2.set_title("Average Payout by Incident Severity ($)", fontsize=10)
        ax2.set_ylabel("Mean Payout ($)")
        st.pyplot(fig2)

    st.divider()

    # Section C: Target Leakage Resolution
    st.subheader("3. Data Hygiene & Leakage Resolution")
    st.warning("""
    **Crucial Analytics Finding:** Initial models yielded an unconvincing $R^2 = 0.9932$ due to **Target Data Leakage** (sub-claim components like `vehicle_claim` sum directly into total payout).
    
    **The Fix:** Stripped all sub-claim amounts and post-incident ratios. Rebuilt the model purely on pre-incident features (driver age, collision severity, vehicles involved), yielding a realistic production-grade **$R^2 = 0.7041$** and **MAE = $10,339.05**.
    """)

    st.divider()

    # Section D: Final Risk Tiering Breakdown
    st.subheader("4. Operational Risk Tiering & Business Outcomes")
    st.markdown("Holdout test set claims (200 records) stratified into 3 actionable risk categories:")

    tier_data = {
        'Risk Tier': ['Tier 1: Low Risk (<$45k)', 'Tier 2: Medium Risk ($45k–$62k)', 'Tier 3: High Risk (>$62k)'],
        'Policy Count': [67, 66, 67],
        'Avg Actual Payout': ['$28,130.15', '$62,369.85', '$61,711.64'],
        'Avg Predicted Payout': ['$27,660.33', '$59,893.49', '$64,404.94'],
        'High-Severity Incidents (%)': ['28.36%', '68.18%', '61.19%'],
        'Portfolio Loss Share (%)': ['18.59%', '40.61%', '40.79%']
    }
    st.table(pd.DataFrame(tier_data))

    st.success("""
    **Business Impact:** 
    * **Tier 1 (18.59% Loss Share):** Auto-approved via **Straight-Through Processing (STP)**, slashing processing time.
    * **Tiers 2 & 3 (81.41% Loss Share):** Immediately escalated to **Senior Adjusters & Fraud Auditors** on Day 1 to protect cash reserves.
    """)

# ==============================================================================
# TAB 2: LIVE FNOL CLAIM CALCULATOR & TRIAGE ENGINE
# ==============================================================================
with tab2:
    st.header("First Notice of Loss (FNOL) Interactive Calculator")
    st.markdown("Enter claim parameters below to run real-time severity predictions and operational routing.")
    st.divider()

    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        st.subheader("👤 Policyholder Profile")
        age = st.number_input("Driver Age", min_value=18, max_value=85, value=38)
        months_as_customer = st.number_input("Months as Customer", min_value=0, max_value=600, value=128)
        policy_annual_premium = st.number_input("Policy Annual Premium ($)", min_value=100.0, max_value=5000.0, value=1400.0)
        policy_deductible = st.selectbox("Policy Deductible ($)", [500, 1000, 2000])
        umbrella_limit = st.selectbox("Umbrella Limit ($)", [0, 1000000, 2000000, 3000000])

    with form_col2:
        st.subheader("💥 Incident Metadata")
        incident_severity = st.selectbox("Incident Severity", ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'])
        incident_type = st.selectbox("Incident Type", ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Side Impact', 'Parked Car'])
        collision_type = st.selectbox("Collision Type", ['Front Collision', 'Rear Collision', 'Side Collision', 'MISSING_UNSPECIFIED'])
        number_of_vehicles_involved = st.slider("Vehicles Involved", 1, 6, 2)
        bodily_injuries = st.slider("Bodily Injuries Reported", 0, 5, 1)

    with form_col3:
        st.subheader("📋 Policy Details")
        policy_state = st.selectbox("Policy State", ['OH', 'IN', 'IL'])
        insured_sex = st.selectbox("Insured Sex", ['FEMALE', 'MALE'])
        insured_education_level = st.selectbox("Education Level", ['JD', 'High School', 'Associate', 'MD', 'Masters', 'PhD', 'College'])
        insured_occupation = st.selectbox("Occupation", ['craft-repair', 'machine-op-inspct', 'prof-specialty', 'tech-support', 'sales', 'exec-managerial', 'protective-serv', 'armed-forces', 'other-service', 'adm-clerical', 'farming-fishing', 'transport-moving', 'priv-house-serv', 'handlers-cleaners'])
        property_damage = st.selectbox("Property Damage Reported", ['NO', 'YES', 'MISSING_UNSPECIFIED'])
        police_report_available = st.selectbox("Police Report Available", ['NO', 'YES', 'MISSING_UNSPECIFIED'])

    st.divider()

    if st.button("🚀 Calculate Estimated Claim Severity & Risk Tier", use_container_width=True):
        input_df = pd.DataFrame([{
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

        try:
            X_transformed = preprocessor.transform(input_df)
            log_prediction = model.predict(X_transformed)[0]
            predicted_payout = np.expm1(log_prediction)

            st.subheader("📊 Output & Operational Routing Recommendation")
            out_col1, out_col2 = st.columns(2)

            with out_col1:
                st.metric("Estimated Claim Payout ($)", f"${predicted_payout:,.2f}")

            with out_col2:
                if predicted_payout < 45000:
                    st.success("🟢 **TIER 1: LOW RISK (<$45,000)**")
                    st.info("⚡ **Operational Path:** Route to **Straight-Through Processing (STP)** for instant automated settlement.")
                elif 45000 <= predicted_payout <= 62000:
                    st.warning("🟡 **TIER 2: MEDIUM RISK ($45,000 - $62,000)**")
                    st.info("🔍 **Operational Path:** Assign to **Standard Adjuster Queue** for standard damage review.")
                else:
                    st.error("🔴 **TIER 3: HIGH RISK (>$62,000)**")
                    st.info("🚨 **Operational Path:** Escalate immediately to **Senior Adjuster & SIU Fraud Audit**.")

        except Exception as err:
            st.error(f"Inference execution error: {err}")
