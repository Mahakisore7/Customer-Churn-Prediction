import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading and Preprocessing Data...")
df = pd.read_csv(r'Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv') # Ensure path is correct

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)


binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})


df = pd.get_dummies(df)


scaler = MinMaxScaler()
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


X = df.drop('Churn', axis=1).values
y = df['Churn'].values


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Data Ready! Total samples: {X_resampled.shape[0]}")

# ==========================================
# PART 2: LIBRARY KNN IMPLEMENTATION
# ==========================================

# 1. Split Data into Train (80%) and Test (20%)
# random_state=42 ensures we get the same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("\n--------------------------------")
print(f"Training Set Size: {X_train.shape[0]} customers")
print(f"Testing Set Size:  {X_test.shape[0]} customers")
print("--------------------------------")

# 2. Initialize the Model
# n_neighbors=5 is the standard starting point (K=5)
knn_model = KNeighborsClassifier(n_neighbors=5)


print("\nTraining the model (this might take a second)...")
knn_model.fit(X_train, y_train)


print("Predicting on Test Data...")
y_pred = knn_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\n================================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("================================")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 1. Calculate the Matrix numbers
cm = confusion_matrix(y_test, y_pred)

# 2. Create the Heatmap Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted No (Stay)', 'Predicted Yes (Churn)'],
            yticklabels=['Actual No (Stay)', 'Actual Yes (Churn)'])

# 3. Add Labels and Title
plt.title('Confusion Matrix - Library KNN')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 4. Show the Image
plt.show()