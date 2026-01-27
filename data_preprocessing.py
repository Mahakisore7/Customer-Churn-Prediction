import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE  


df = pd.read_csv(r'Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. CLEANING
# Force TotalCharges to be numeric, turn errors (empty strings) into NaNs, then fill with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
# Drop customerID (useless for prediction)
df = df.drop('customerID', axis=1)

# 3. ENCODING
# Convert Binary Yes/No to 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# One-Hot Encode the rest (InternetService, PaymentMethod, etc.)
df = pd.get_dummies(df)

# 4. SCALING
# Scale numerical columns to 0-1 range
scaler = MinMaxScaler()
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# 5. SPLIT & SMOTE
X = df.drop('Churn', axis=1).values  
y = df['Churn'].values

# Handle Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Data is ready!")
print(f"Features (X) Shape: {X_resampled.shape}")
print(f"Target (y) Shape: {y_resampled.shape}")