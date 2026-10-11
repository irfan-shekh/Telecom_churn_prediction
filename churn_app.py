import os
import joblib
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a polished UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📡 Telecom Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time risk scoring and proactive retention analytics</div>', unsafe_allow_html=True)

# Load Model
MODEL_PATH = "random_forest_model.joblib"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_trained_model()

# Sidebar for metadata / quick settings
with st.sidebar:
    st.header("About This App")
    st.info(
        "This tool predicts customer churn risk using customer tenure, "
        "contract commitment, internet service tier, and financial charges."
    )
    st.markdown("---")
    st.caption("⚡ Model: Random Forest Classifier")
    st.caption("🎯 Primary Objective: Proactive Churn Intervention")

# Main Interface: Two-column layout
col_input, col_result = st.columns([1.1, 1], gap="large")

with col_input:
    st.subheader("📋 Customer Account Profile")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=6,
                step=1,
                help="Number of months customer has stayed with the company.",
            )
            contract = st.selectbox(
                "Contract Type",
                options=["Month-to-month", "One year", "Two year"],
                help="Longer contracts significantly lower churn probability.",
            )
        with c2:
            internet_service = st.selectbox(
                "Internet Service Tier",
                options=["DSL", "Fiber optic", "No"],
                help="Fiber optic subscribers often show higher price sensitivity.",
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=15.0,
                max_value=150.0,
                value=70.0,
                step=5.0,
            )

        # Estimate Total Charges if not provided
        default_total = float(round(tenure * monthly_charges, 2))
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=default_total if default_total > 0 else 70.0,
            step=25.0,
            help="Total amount billed to the customer so far.",
        )

        predict_button = st.button("🔮 Calculate Churn Risk", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Risk Assessment & Recommendation")

    # Mapping inputs to expected numerical representation
    label_mapping_internet = {"DSL": 0, "Fiber optic": 1, "No": 2}
    label_mapping_contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}

    input_df = pd.DataFrame(
        [
            {
                "tenure": tenure,
                "InternetService": label_mapping_internet[internet_service],
                "Contract": label_mapping_contract[contract],
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": float(total_charges),
            }
        ]
    )

    # Perform inference
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    churn_proba = probabilities[1]
    stay_proba = probabilities[0]

    # Display Metrics & Probability Bar
    st.markdown("#### Churn Probability")
    st.progress(float(churn_proba))
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Churn Risk", f"{churn_proba * 100:.1f}%")
    with m2:
        st.metric("Retention Likelihood", f"{stay_proba * 100:.1f}%")

    st.markdown("---")

    # Business Decision Logic
    if churn_proba >= 0.50:
        st.error("⚠️ **High Churn Risk: Customer is Likely to Leave**")
        st.markdown(
            """
            **Recommended Retention Actions:**
            - **Contract Upgrade:** Offer an onboarding incentive or loyalty credit to transition from *Month-to-month* to an annual contract.
            - **Value Audit:** Review whether high monthly charges align with service usage.
            - **Priority Support:** Initiate proactive outreach from a customer success manager.
            """
        )
    elif churn_proba >= 0.30:
        st.warning("⚡ **Moderate Risk: Monitor Account**")
        st.markdown(
            """
            **Recommended Action:**
            - Check customer satisfaction and send targeted promotional add-ons (e.g. Free TechSupport or Security trial).
            """
        )
    else:
        st.success("✅ **Low Churn Risk: Customer is Stable & Loyal**")
        st.markdown(
            """
            **Customer Status:**
            - Healthy account profile. Ideal candidate for long-term loyalty rewards or cross-sell opportunities.
            """
        )
