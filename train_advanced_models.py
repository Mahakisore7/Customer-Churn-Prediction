# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.svm import SVC
# from xgboost import XGBClassifier


# print("1. Loading and Preprocessing Data...")
# df = pd.read_csv(r'Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv') 


# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
# df = df.drop('customerID', axis=1)

# binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
# for col in binary_cols:
#     df[col] = df[col].map({'Yes': 1, 'No': 0})
# df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
# df = pd.get_dummies(df)


# scaler = MinMaxScaler()
# cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
# df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


# X = df.drop('Churn', axis=1).values
# y = df['Churn'].values

# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# print(f"Data Ready. Training Size: {len(X_train)} | Test Size: {len(X_test)}")

# # ==========================================
# # PART 2: SUPPORT VECTOR MACHINE (SVM)
# # ==========================================
# print("\n------------------------------------------------")
# print("TRAINING MODEL 2: SVM (Geometric Approach)")
# print("------------------------------------------------")
# # kernel='linear' draws a straight line. 
# # kernel='rbf' (default) draws a curvy line (better for complex data).
# svm_model = SVC(kernel='rbf', random_state=42)
# svm_model.fit(X_train, y_train)

# y_pred_svm = svm_model.predict(X_test)
# acc_svm = accuracy_score(y_test, y_pred_svm)

# print(f"SVM Accuracy: {acc_svm * 100:.2f}%")
# print("SVM Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_svm))

# # ==========================================
# # PART 3: XGBOOST (Gradient Boosting)
# # ==========================================
# print("\n------------------------------------------------")
# print("TRAINING MODEL 3: XGBoost (Ensemble Approach)")
# print("------------------------------------------------")
# # use_label_encoder=False removes a warning
# # eval_metric='logloss' is standard for binary classification
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# xgb_model.fit(X_train, y_train)

# y_pred_xgb = xgb_model.predict(X_test)
# acc_xgb = accuracy_score(y_test, y_pred_xgb)

# print(f"XGBoost Accuracy: {acc_xgb * 100:.2f}%")
# print("XGBoost Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_xgb))

# # ==========================================
# # PART 4: FINAL COMPARISON
# # ==========================================
# print("\n================================================")
# print(f"FINAL LEADERBOARD")
# print(f"1. XGBoost Accuracy:    {acc_xgb * 100:.2f}%")
# print(f"2. SVM Accuracy:        {acc_svm * 100:.2f}%")
# print(f"3. Manual KNN (Prev):   80.05%") 
# print("================================================")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ==========================================
# 1. MANUAL KNN CLASS (FROM SCRATCH)
# ==========================================
class ManualKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)

    def predict(self, X_test):
        predictions = []
        X_test = np.array(X_test, dtype=float)
        
        # Optimization: Process in chunks if data is huge, but for 2000 rows, this loop is fine
        for i, x_query in enumerate(X_test):
            # Vectorized Euclidean Distance
            distances = np.sqrt(np.sum((self.X_train - x_query)**2, axis=1))
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[idx] for idx in k_indices]
            
            # Majority Vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
            # Progress tracker
            if (i+1) % 500 == 0:
                print(f"   [Manual KNN] Processed {i+1} / {len(X_test)} samples...")
                
        return np.array(predictions)

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print("--- Phase 1: Data Engineering ---")
df = pd.read_csv(r'Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)

# Binary Encoding
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# One-Hot Encoding
df = pd.get_dummies(df)

# Scaling
scaler = MinMaxScaler()
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Split & SMOTE
X = df.drop('Churn', axis=1).values
y = df['Churn'].values

print("   Applying SMOTE (Balancing)...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(f"   Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

# ==========================================
# 3. TRAINING & PREDICTION
# ==========================================

# --- Model A: Manual KNN ---
print("\n--- Phase 2: Training Manual KNN (k=5) ---")
knn_manual = ManualKNN(k=5)
knn_manual.fit(X_train, y_train)
y_pred_knn = knn_manual.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

# --- Model B: SVM ---
print("\n--- Phase 3: Training SVM (RBF Kernel) ---")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# --- Model C: XGBoost ---
print("\n--- Phase 4: Training XGBoost ---")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# ==========================================
# 4. REPORT GENERATION
# ==========================================
print("\n================================================")
print(f"FINAL LEADERBOARD")
print(f"1. XGBoost:     {acc_xgb * 100:.2f}%")
print(f"2. SVM:         {acc_svm * 100:.2f}%")
print(f"3. Manual KNN:  {acc_knn * 100:.2f}%")
print("================================================")

print("\n--- Classification Report: Manual KNN ---")
print(classification_report(y_test, y_pred_knn))

print("\n--- Classification Report: SVM ---")
print(classification_report(y_test, y_pred_svm))

print("\n--- Classification Report: XGBoost ---")
print(classification_report(y_test, y_pred_xgb))

# ==========================================
# 5. VISUALIZATION (3 MATRIX PLOTS)
# ==========================================
print("\nGenerating Comparison Heatmaps...")

# Calculate Matrices
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Setup Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Helper to plot each axis
def plot_cm(ax, cm, title, color):
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'], cbar=False)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')

# Plot 1: KNN
plot_cm(axes[0], cm_knn, f'Manual KNN\nAcc: {acc_knn*100:.2f}%', 'Purples')

# Plot 2: SVM
plot_cm(axes[1], cm_svm, f'SVM (Geometric)\nAcc: {acc_svm*100:.2f}%', 'Blues')

# Plot 3: XGBoost
plot_cm(axes[2], cm_xgb, f'XGBoost (Ensemble)\nAcc: {acc_xgb*100:.2f}%', 'Greens')

plt.tight_layout()
plt.show()
