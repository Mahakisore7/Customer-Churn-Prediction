import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns

import matplotlib.pyplot as plt
# ==========================================
# PART 1: THE MANUAL KNN CLASS 
# ==========================================
class ManualKNN:
    def __init__(self, k=5):
        """
        Store the value of k (e.g., 5 neighbors).
        """
        self.k = k

    def fit(self, X, y):
        """
        KNN is a 'Lazy Learner'. It doesn't actually learn a formula.
        It just memorizes the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        """
        Predict labels for the Test data.
        """
        predictions = []
        
        # Loop through each customer in the test set
        for i, x_query in enumerate(X_test):
            
            # --- THE MATHEMATICAL CORE (Vectorized Distance) ---
            
            # 1. Calculate Euclidean Distance between this customer (x_query)
            #    and ALL training customers (self.X_train) at once.
            #    Formula: sqrt( sum( (x_train - x_query)^2 ) )
            distances = np.sqrt(np.sum((self.X_train - x_query)**2, axis=1))
            
            # 2. Get indices of the k nearest neighbors
            #    np.argsort sorts the distances from small to large
            #    We take the first k indices.
            k_indices = np.argsort(distances)[:self.k]
            
            # 3. Get the labels (Churn/No Churn) of these neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # 4. Vote! (Majority Rule)
            #    Counter counts the votes: {0: 2, 1: 3} -> Churn wins
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
            #  Print progress every 500 customers
            if (i+1) % 500 == 0:
                print(f"Processed {i+1} / {len(X_test)} customers...")
                
        return np.array(predictions)

# ==========================================
# PART 2: TESTING THE MODEL
# ==========================================
if __name__ == "__main__":
    print("1. Loading and Preparing Data...")
    
    
    df = pd.read_csv(r'Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv') 
    

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
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    
    print("\n2. Initializing Manual KNN (k=5)...")
    manual_model = ManualKNN(k=5)
    
    print("3. Training (Memorizing Data)...")
    manual_model.fit(X_train, y_train)
    
    print("4. Predicting (Calculating Distances)...")
    y_pred_manual = manual_model.predict(X_test)
   
    acc = accuracy_score(y_test, y_pred_manual)
    print(f"\nManual KNN Accuracy: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_manual))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_manual))
    # 1. Calculate the Matrix numbers
    cm = confusion_matrix(y_test, y_pred_manual)

    # 2. Create the Heatmap Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted No (Stay)', 'Predicted Yes (Churn)'],
                yticklabels=['Actual No (Stay)', 'Actual Yes (Churn)'])

    # 3. Add Labels and Title
    plt.title('Confusion Matrix - Manual KNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 4. Show the Image
    plt.show()