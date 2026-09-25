# GuardURL — Model Training & Evaluation Summary

## Experimental Configuration
- **Dataset:** PhiUSIIL Phishing URL Dataset
- **Total Samples Evaluated:** 99,999
- **Train Split (80%):** 79,999 samples
- **Test Split (20%):** 20,000 unseen samples
- **Feature Count:** 34 URL-based features
- **Random Seed:** 42 (Fully Reproducible)

## Verified Evaluation Results on Unseen Test Split

| Model | Accuracy | Phishing Recall | Phishing F1 | Phishing FNR | Macro F1 | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9972 | 0.9943 | 0.9968 | 0.0057 | 0.9972 | 0.9990 | 0.4s |
| **Random Forest** | 0.9966 | 0.9937 | 0.9960 | 0.0063 | 0.9965 | 0.9992 | 0.56s |
| **XGBoost** | 0.9973 | 0.9945 | 0.9968 | 0.0055 | 0.9972 | 0.9993 | 0.39s |

## Model Selection Decision
- **Winning Model:** **XGBoost**
- **Selection Criterion:** Maximization of Phishing Detection Recall (minimizing False Negative Rate) combined with Phishing F1-score and overall accuracy.
- **Saved Model Location:** `models/best_model.joblib`
- **Preprocessor Location:** `models/preprocessor.joblib`
