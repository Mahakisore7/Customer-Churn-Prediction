import pandas as pd
import joblib  # This is the tool to save files
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==========================================
# 1. LOAD & PREPROCESS (The same as before)
# ==========================================
print("Loading data...")
df = pd.read_csv(r'Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)

# Encode Binary
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# One-Hot Encode (Get Dummies)
df = pd.get_dummies(df)

# ==========================================
# 2. SAVE THE COLUMN NAMES
# ==========================================
# We need to know exactly which columns the model expects (e.g., 'InternetService_DSL')
# We drop 'Churn' because that's the target, not a feature.
model_columns = list(df.drop('Churn', axis=1).columns)
joblib.dump(model_columns, 'model_columns.pkl')
print(f"Saved {len(model_columns)} column names to 'model_columns.pkl'")

# ==========================================
# 3. SCALE & SAVE THE SCALER
# ==========================================
scaler = MinMaxScaler()
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Fit the scaler on the data
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Save the scaler so we can scale new users exactly the same way later
joblib.dump(scaler, 'scaler.pkl')
print("Saved Scaler to 'scaler.pkl'")

# ==========================================
# 4. TRAIN & SAVE THE MODEL
# ==========================================
X = df.drop('Churn', axis=1).values
y = df['Churn'].values

# SMOTE (Balancing)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Training Final XGBoost Model...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_resampled, y_resampled)

# Save the trained model
joblib.dump(xgb_model, 'final_churn_model.pkl')
print("Saved XGBoost Model to 'final_churn_model.pkl'")
print("\nSuccess! You are ready for deployment.")