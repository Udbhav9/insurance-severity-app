# Auto Insurance Claim Severity Prediction & Loss Reserving Engine

An end-to-end predictive analytics platform and web application designed to estimate auto insurance claim payout severity at **First Notice of Loss (FNOL)**. By leveraging strictly pre-incident policyholder and crash metadata, the engine automates operational claim triage and informs capital loss-reserving decisions.

---

## Business Problem & Operational Context

When an auto insurance claim is reported on Day 1 (First Notice of Loss), insurers face immediate financial uncertainty: final repair bills, medical expenses, and legal settlements often take weeks or months to materialize. Treating all incoming claims equally creates two major operational bottlenecks:

1. **Inaccurate Loss Reserving:** Actuaries and finance teams struggle to allocate regulatory cash reserves accurately without an early severity estimate.
2. **Manual Queue Congestion:** Routine, low-cost claims sit in the same review queues as complex, high-exposure crashes, driving up administrative labor costs and slowing down customer payouts.

### Solution Overview
This project delivers a leak-free **XGBoost Regression & Risk Triage Engine** deployed via Streamlit. It predicts final claim payout amounts immediately upon claim intake and routes policies into three actionable operational tiers.

---

## Dataset Overview

* **Source:** Kaggle Auto Insurance Claims Dataset (`buntyshah/auto-insurance-claims-data`)
* **Volume:** 1,000 policyholder claim records
* **Target Variable:** `total_claim_amount` (Mean payout: **~$52,761.94**; Total portfolio exposure: **$52.76M**)
* **Feature Categories (Pre-Incident Only):**
  * **Policyholder Profile:** Driver age, months as customer, engineered `customer_tenure_ratio`, education level, occupation, state.
  * **Policy Coverage Terms:** Annual premium, policy deductible, umbrella limit, combined single limit (`policy_csl`).
  * **Incident Metadata:** Incident severity (*Trivial*, *Minor*, *Major*, *Total Loss*), incident type, collision type, number of vehicles involved, bodily injuries reported, property damage, police report availability, and witnesses.

---

## Analytical Methodology & Feature Hygiene

1. **Target Data Leakage Prevention:**
   * Post-incident sub-claim breakdowns (`injury_claim`, `property_claim`, `vehicle_claim`) directly sum to the target payout and are unavailable at FNOL. All post-incident financial variables were purged so the model relies strictly on Day-1 intake data.
2. **Target Distribution Normalization:**
   * Raw claim payouts exhibit substantial right-skewness. A logarithmic transformation ($\log(1 + x)$) was applied during training to stabilize variance, with predictions exponentiated back to dollar values ($\exp(y) - 1$) at inference.
3. **Preprocessing Pipeline:**
   * Built a `ColumnTransformer` pipeline utilizing `StandardScaler` for numerical predictors and `OneHotEncoder(drop='first', handle_unknown='ignore')` for categorical features.
4. **Predictive Modeling:**
   * Trained an **XGBoost Regressor** to capture non-linear interactions between incident scale, vehicle count, and coverage limits.

---

## Model Performance & Key Findings

Evaluated on a 20% holdout test set (200 claims):

* **Coefficient of Determination ($R^2$):** `0.7041`
* **Mean Absolute Error (MAE):** `~$10,339.05`
* **Primary Cost Driver:** `incident_severity` (*Major Damage* and *Total Loss*) accounts for over 80% of total dollar exposure across the portfolio.

### Operational Risk Tiering Outcomes

Test predictions were stratified into three operational risk tiers to guide claims routing:

| Risk Tier | Predicted Payout Range | Avg Actual Payout | Portfolio Loss Share | Operational Action |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Low Risk** | `< $45,000` | `$28,130.15` | **18.59%** | **Straight-Through Processing (STP):** Auto-approve via mobile app for instant settlement. |
| **Tier 2: Medium Risk** | `$45,000 – $62,000` | `$62,369.85` | **40.61%** | **Standard Adjuster Queue:** Route for standard repair & estimate verification. |
| **Tier 3: High Risk** | `> $62,000` | `$61,711.64` | **40.79%** | **Senior Adjuster & SIU Audit:** Escalate immediately on Day 1 to lock in reserves and audit fraud risk. |

> **Key Business Takeaway:** Tiers 2 and 3 represent **81.40% of total dollar loss exposure**. Flagging these claims at FNOL protects capital reserves, while automating Tier 1 eliminates manual processing overhead on low-risk claims.

---

## Tech Stack

* **Language:** Python 3.10+
* **Data Manipulation & Preprocessing:** Pandas, NumPy, Scikit-Learn
* **Machine Learning:** XGBoost
* **Data Visualization:** Matplotlib, Seaborn
* **Web Application & Deployment:** Streamlit Community Cloud, GitHub

---

## Repository Structure

```text
insurance-severity-app/
├── app.py                  # 2-tab Streamlit application (Executive Analysis + Live FNOL Calculator)
├── X_clean.csv             # Pre-processed feature schema dataset for live pipeline fitting
├── xgb_model.json          # Native JSON-serialized XGBoost regression model
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## Local Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/insurance-severity-app.git
   cd insurance-severity-app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```