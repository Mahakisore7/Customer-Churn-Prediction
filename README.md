# 📡 Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-orange)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success)](https://github.com)

**Team 11 | Machine Learning Project**

---

## 🎯 Problem Statement

Predict which customers will stop using the service (churn) to enable proactive retention strategies. Customer acquisition costs 5-25x more than retention, making churn prediction critical for business sustainability.

### Objective
Minimize **False Negatives** (missing churning customers) using a custom Manual KNN implementation and benchmark it against industry-standard algorithms like XGBoost.

---

## 🚀 Project Overview

This project builds a robust predictive system to identify high-risk customers. We implement algorithms **from scratch** to demonstrate deep understanding of the mathematics behind instance-based learning and compare them with state-of-the-art ensemble methods.

### Key Features

- ✅ **Manual Implementation**: K-Nearest Neighbors (KNN) built from scratch using NumPy (vectorized)
- ✅ **Advanced Benchmarking**: Comparison with SVM (geometric) and XGBoost (gradient boosting)
- ✅ **Data Engineering**: Handling imbalanced data (26% churn) using SMOTE and custom scaling pipelines
- ✅ **Mathematical Rigor**: Euclidean distance computation via NumPy broadcasting for efficiency

---

## 📊 Methodology

### Phase 1: Data Engineering

**Dataset**: Telco Customer Churn
- **Samples**: 7,043 customers
- **Features**: 21 original features (expanded to 40 after encoding)
- **Target**: Binary classification (Churn: Yes/No)

**Preprocessing Pipeline**:

1. **Cleaning**: Handled missing values in `TotalCharges` column
2. **Encoding**: One-Hot Encoding for categorical variables
3. **Scaling**: Min-Max Scaling for numerical features (`Tenure`, `MonthlyCharges`, `TotalCharges`)
   - Critical for distance-based models like KNN
4. **Balancing**: Applied SMOTE to address class imbalance (73% non-churn, 27% churn)

---

### Phase 2: Manual KNN (From Scratch)

Implemented a custom `ManualKNN` class using pure NumPy.

**Mathematical Foundation**:

The Euclidean distance between points **p** and **q** is calculated as:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

**Implementation Highlights**:
- Vectorized distance calculation using NumPy broadcasting
- Efficient computation across 8,000+ training samples
- Majority voting mechanism for class prediction
- Configurable K parameter for neighbor selection

---

### Phase 3: Model Benchmarking

We evaluated three distinct algorithmic families:

| Model | Algorithm Type | Key Characteristic |
|-------|---------------|-------------------|
| **Manual KNN** | Instance-based learning | Custom implementation (baseline) |
| **SVM (RBF Kernel)** | Geometric boundary detection | Non-linear decision boundaries |
| **XGBoost** | Gradient boosting ensemble | State-of-the-art performance |

---

## 📈 Results

### Performance Leaderboard

| Rank | Model | Accuracy | Recall (Churn Class) |
|------|-------|----------|---------------------|
| 🥇 | **XGBoost** | **85.60%** | High |
| 🥈 | **SVM (RBF)** | **82.17%** | Moderate |
| 🥉 | **Manual KNN** | **80.05%** | Moderate |

### Analysis

- **XGBoost** outperformed the manual implementation by ~5.5% due to its ability to capture non-linear feature interactions that simple distance metrics miss
- **Manual KNN** achieved 80% accuracy, validating the custom mathematical implementation
- The gap between models highlights the trade-off between interpretability (KNN) and performance (XGBoost)

---

## 🛠️ Installation & Usage

### 1. Prerequisites

```bash
# Install required packages
pip install pandas numpy scikit-learn imbalanced-learn xgboost
```

**Requirements**:
- Python 3.8+
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Scikit-learn >= 0.24.0
- Imbalanced-learn >= 0.8.0
- XGBoost >= 1.4.0

---

### 2. Data Preprocessing

Clean the data and generate the processed dataset:

```bash
python data_preprocessing.py
```

**Outputs**:
- Cleaned dataset with encoded features
- SMOTE-balanced training set
- Scaled numerical features

---

### 3. Run Manual KNN

Test the custom "from scratch" implementation:

```bash
python manual_knn.py
```

**What it does**:
- Loads preprocessed data
- Trains custom KNN classifier
- Evaluates performance metrics
- Displays confusion matrix

---

### 4. Run Benchmark Models

Compare SVM and XGBoost performance:

```bash
python train_advanced_models.py
```

**What it does**:
- Trains SVM with RBF kernel
- Trains XGBoost classifier
- Generates comparative performance report
- Saves model artifacts

---

## 📂 Project Structure

```
churn-prediction/
│
├── Dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Raw dataset
│
├── data_preprocessing.py                        # Data cleaning, encoding, SMOTE
├── manual_knn.py                                # Custom KNN implementation (NumPy)
├── train_library_knn.py                         # Sklearn KNN (sanity check)
├── train_advanced_models.py                     # SVM & XGBoost training
│
├── requirements.txt                             # Python dependencies
└── README.md                                    # Project documentation
```

---

## 🧮 Technical Details

### Manual KNN Implementation

```python
class ManualKNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X_train, y_train):
        # Store training data
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        # Vectorized Euclidean distance calculation
        distances = np.sqrt(((X_test[:, np.newaxis] - self.X_train) ** 2).sum(axis=2))
        
        # Find K nearest neighbors
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        
        # Majority voting
        k_labels = self.y_train[k_indices]
        predictions = np.array([np.bincount(labels).argmax() for labels in k_labels])
        
        return predictions
```

---

## 👥 Team Members

**Team 11**

| Name | Student ID |
|------|-----------|
| **Hemanth S.N** | CB.SC.U4AIE24321 |
| **Mahakisore M** | CB.SC.U4AIE24333 |
| **Yashwanth B** | CB.SC.U4AIE24360 |

---

## 📚 References

1. Customer acquisition vs retention costs - Harvard Business Review
2. Telco Customer Churn Dataset - IBM Sample Data Sets
3. Feature scaling for machine learning - Scikit-learn documentation
4. SMOTE: Synthetic Minority Over-sampling Technique - Journal of Artificial Intelligence Research

---

## 📝 License

This project is part of an academic assignment and is intended for educational purposes only.

---

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the team members listed above.

---

## 🌟 Acknowledgments

- Dataset provided by IBM Sample Data Sets
- Course instructors for guidance on machine learning fundamentals
- Open-source community for tools like NumPy, Scikit-learn, and XGBoost

---

**Made with ❤️ by Team 11**
