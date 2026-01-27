import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the App
app = FastAPI(title="Telecom Churn Prediction API")

# 2. Load the Saved Files
# These are the files you just created with save_model.py
print("Loading Model Components...")
model = joblib.load('final_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl') # We need this to remember the 40 columns order
print("Model Loaded Successfully!")

# 3. Define the Input Format (The "Customer Profile")
# This ensures the user sends the right data types
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 4. The Prediction Endpoint (The "Brain")
@app.post("/predict")
def predict_churn(customer: Customer):
    # A. Convert Input JSON to DataFrame
    input_data = customer.dict()
    df = pd.DataFrame([input_data])
    
    # B. PREPROCESSING (Must match Training Phase exactly!)
    
    # 1. Binary Encoding (Yes/No -> 1/0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Gender (Female/Male -> 1/0) - Note: In training we mapped Female:1
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    
    # 2. One-Hot Encoding
    # This is tricky! If input is "DSL", get_dummies makes 'InternetService_DSL'.
    # But the model also expects 'InternetService_Fiber' (which is missing here).
    df = pd.get_dummies(df)
    
    # 3. Align with Training Columns (The Critical Fix)
    # We force the DataFrame to have exactly the same 40 columns as training.
    # Any missing column (like 'InternetService_Fiber') gets filled with 0.
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # 4. Scaling
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # C. PREDICT
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    # D. RETURN RESULT
    churn_risk = "High Risk (Churn)" if prediction[0] == 1 else "Low Risk (Stay)"
    confidence = float(probability[0][1]) # Probability of being class 1 (Churn)
    
    return {
        "prediction": churn_risk,
        "churn_probability": f"{confidence * 100:.2f}%"
    }

# 5. Run Instructions
# To run this: uvicorn app:app --reload