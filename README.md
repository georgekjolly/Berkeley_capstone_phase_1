# 💳 Credit Card Fraud Detection

**Author**: George Kareemadom Jolly  
**Project Type**: Supervised Classification (Imbalanced Binary Classification)  
**Model Used**: XGBoost
**Notebook link**: https://github.com/georgekjolly/Berkeley_capstone_phase_1/blob/main/capstone-phase-1_v2.ipynb

---

## 🧠 Executive Summary

Credit card fraud poses a significant risk to financial institutions and customers. Detecting fraud early helps reduce financial losses and maintain trust in digital banking systems. This project aims to build an accurate and robust machine learning model to detect fraudulent credit card transactions using transactional and customer metadata. Using XGBoost and relevant preprocessing techniques, the model achieves strong recall and AUC, which are critical for identifying rare fraud events in highly imbalanced data.

---

## ❓ Rationale

Fraudulent credit card transactions cause substantial losses globally and damage trust in the financial system. Manual detection is ineffective and unscalable due to the large volume of transactions. A machine learning solution can help financial institutions:
- Automatically detect fraudulent activity
- Reduce false negatives (missed fraud)
- Respond quickly to potential threats

---

## 🎯 Research Question

> Can a machine learning model accurately identify fraudulent transactions in a highly imbalanced dataset?

This question focuses on improving fraud detection rates while minimizing false positives, ensuring the model is practical and efficient in real-world applications.

---

## 📊 Data Sources

This project uses a real-world-like dataset of credit card transactions, containing:
- Transaction metadata (amount, time, merchant, etc.)
- Cardholder details (anonymized)
- Label: `isFraud` (1 = fraud, 0 = legitimate)

Fields include:
- Categorical: `merchantName`, `merchantCategoryCode`, `acqCountry`, etc.
- Numerical: `transactionAmount`, `creditLimit`, `availableMoney`, etc.
- Date fields: `transactionDateTime`, `accountOpenDate`, etc.

---

## 🛠️ Methodology

### Data Preprocessing
- Convert dates to datetime format
- Extract time-based features: hour, weekday, month, etc.
- Calculate account age and time since last address change
- One-hot encode categorical features
- Apply smoothed target encoding on high-cardinality categorical columns
- Label encode nominal variables like `transactionType`
- Handle missing values
- Remove multicollinearity between highly correlated features

### Handling Imbalance
- Acknowledge the highly imbalanced nature of the data (fraud < 2%)
- Focus on metrics like recall, precision, and ROC AUC over accuracy
- Optionally experiment with oversampling or class-weighted models

### Model Building
- Use XGBoost as the primary classifier
- Train on the processed dataset with stratified splits
- Predict on unseen test data

### Evaluation Metrics
- Confusion matrix
- Precision, Recall, F1-Score
- ROC AUC Score

---

## 📈 Results

- **Recall (fraud cases)**: 72% — captures the majority of fraudulent transactions
- **Precision (fraud cases)**: ~5% — due to the rarity of fraud in the dataset
- **F1 Score**: Balanced evaluation for imbalanced classification
- **ROC AUC Score**: 0.82 — very good discrimination between fraud and legitimate
- **Overall Accuracy**: 78% — skewed due to imbalance, not a priority metric

These results demonstrate that the model performs well in identifying rare fraudulent transactions while maintaining reasonable precision.

---

## 💬 Discussion & Next Steps

### Observations
- Time-based features (hour, day) and merchant info are highly predictive
- Imbalance requires careful evaluation metric selection (recall, AUC preferred)
- False positives are tolerable in fraud detection compared to false negatives

### Future Improvements
- Tune hyperparameters using grid/random search or Bayesian optimization
- Test ensemble models (e.g., stacking, bagging)
- Try deep learning methods (e.g., autoencoders, LSTM for sequences)
- Deploy model in a streaming pipeline with real-time alerts
- Use active learning or feedback loops to retrain with new data

---

