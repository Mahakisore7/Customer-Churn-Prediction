import streamlit as st
import requests
import json

# 1. Page Configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# 2. Header Section
st.title("📡 Telecom Customer Retention Dashboard")
st.markdown("### Powered by Team 11 | XGBoost Engine")
st.markdown("---")

# 3. Sidebar for Input (The "Control Panel")
st.sidebar.header("📝 Customer Profile")
st.sidebar.markdown("Enter customer details below to predict churn risk.")

# --- DEMOGRAPHICS ---
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

# --- SERVICES ---
st.sidebar.subheader("Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# --- ACCOUNT INFO ---
st.sidebar.subheader("Account Info")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

# 4. Main Panel - Prediction Logic
if st.sidebar.button("🚀 Predict Churn Risk"):
    
    # Prepare the data dictionary (matches the API format)
    input_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Send Request to FastAPI
    try:
        # Note: Ensure your FastAPI is running on port 8000!
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            probability = result['churn_probability']
            
            # Display Result
            st.markdown("### 🔍 Prediction Result")
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                if "High Risk" in prediction:
                    st.error(f"⚠️ {prediction}")
                else:
                    st.success(f"✅ {prediction}")
            
            with col2:
                st.metric(label="Churn Probability", value=probability)
                
            # Explanation / Advice Section
            st.markdown("---")
            st.markdown("### 💡 Recommended Action")
            if "High Risk" in prediction:
                st.write("**⚠️ Critical Alert:** This customer is likely to leave soon.")
                st.write("- **Immediate Action:** Offer a discount or verify if they are facing technical issues.")
                st.write("- **Focus:** Check their *Monthly Charges* and *Tech Support* history.")
            else:
                st.write("**✅ Safe:** This customer is happy.")
                st.write("- **Strategy:** Upsell new features or sign them up for a longer contract.")
                
            # Show the raw JSON for the mentor to see it's real
            with st.expander("See Raw Data Sent to API"):
                st.json(input_data)
                
        else:
            st.error("Error: Could not connect to the Prediction Engine.")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.warning("Make sure your FastAPI server is running! (uvicorn app:app --reload)")

else:
    # Default State
    st.info("👈 Adjust the customer profile in the sidebar and click **Predict**.")
    st.image("https://cdn-icons-png.flaticon.com/512/4144/4144517.png", width=200) # Simple Churn Icon