📌 Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit based on demographic and marketing campaign data. This is a supervised binary classification problem where the target variable indicates whether a customer subscribed to a term deposit.

📊 Dataset Description

The dataset used is the Bank Marketing Dataset from Kaggle.

Total Instances: 11,162

Total Features: 16 input features

Target Variable: deposit (Yes/No)

Type: Binary Classification

Features include:

Demographic features (age, job, marital status, education)

Financial features (balance, loan, housing)

Campaign-related features (duration, campaign, pdays, previous)

The target variable:

Yes → 1

No → 0

🤖 Models Implemented

The following 6 classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbor (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

| ML Model            | Accuracy  | AUC       | Precision | Recall    | F1 Score  | MCC       |
| ------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.826     | 0.907     | 0.828     | 0.799     | 0.813     | 0.651     |
| Decision Tree       | 0.781     | 0.780     | 0.777     | 0.754     | 0.765     | 0.560     |
| KNN                 | 0.817     | 0.880     | 0.820     | 0.787     | 0.803     | 0.633     |
| Naive Bayes         | 0.720     | 0.804     | 0.784     | 0.563     | 0.657     | 0.447     |
| Random Forest       | 0.859     | 0.919     | 0.828     | 0.886     | 0.857     | 0.721     |
| XGBoost             | **0.865** | **0.927** | **0.838** | **0.886** | **0.861** | **0.731** |

| ML Model            | Observation                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Performs well with high AUC. Shows good generalization and balanced performance across metrics.                                 |
| Decision Tree       | Lower performance compared to ensemble methods. Likely overfitting on training data.                                            |
| KNN                 | Good performance but slightly lower than Logistic Regression. Sensitive to scaling and feature distribution.                    |
| Naive Bayes         | Lowest accuracy among all models. Assumption of feature independence likely not valid for this dataset.                         |
| Random Forest       | Strong performance with high F1 and MCC. Handles feature interactions effectively.                                              |
| XGBoost             | Best overall performer. Achieved highest Accuracy, AUC, F1, and MCC. Gradient boosting improves predictive power significantly. |

