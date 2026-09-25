# GuardURL — Explainable and Adaptive AI-Based Phishing Detection System

**GuardURL** is an advanced, production-grade cybersecurity application that detects phishing websites using Explainable AI (XAI) and an Adaptive Supervisory Reinforcement Learning decision layer.

Trained on the **PhiUSIIL Phishing URL Dataset** (235,795 real-world URLs), GuardURL extracts 34 lexical, structural, and behavioral features directly from any URL string, evaluates threat probabilities in milliseconds, generates exact SHAP feature attribution waterwalls, and provides human-understandable security reasoning.

---

## Key Features

1. **High-Speed URL Feature Extraction (<5ms):**
   - 34 comprehensive features computed without unsafe live crawling or scraping:
     - Lexical properties (URL length, domain length, character ratios, letter/digit density)
     - Structural traits (subdomain hierarchy, directory path depth, port anomalies, query parameters)
     - Adversarial indicators (IP address as hostname, hex/percent obfuscation, `@` credential masking, punycode `xn--` homograph lures)
     - Statistical distributions (Shannon entropy, character continuation rate, empirical TLD reputation score)

2. **Empirically Evaluated ML Architecture:**
   - 3 distinct architectures trained and compared on 100,000 stratified samples:
     - **Logistic Regression** (Linear baseline)
     - **Random Forest** (Nonlinear ensemble of 150 estimators)
     - **XGBoost** (Gradient-boosted decision trees — Selected Best Model)
   - Evaluated strictly using genuine metrics: Accuracy, Phishing Recall, Phishing F1, False Negative Rate (FNR), False Positive Rate (FPR), and ROC-AUC.

3. **Explainable AI (SHAP):**
   - TreeExplainer calculates exact local Shapley values for each feature.
   - Highlights whether each feature pushes the prediction toward **LEGITIMATE** or **PHISHING**.
   - Translates mathematical attribution into human-readable cybersecurity analyst factors.

4. **Adaptive Supervisory Decision Layer (RL):**
   - Decoupled Q-Learning / contextual bandit agent that evaluates prediction confidence.
   - Recommends actions: `ACCEPT_INITIAL_PREDICTION`, `PERFORM_ADDITIONAL_ANALYSIS`, or `FLAG_FOR_REVIEW`.
   - The primary classifier operates independently and reliably without RL.

5. **Modern Cybersecurity SOC Dashboard:**
   - Dark-mode glassmorphic interface with real-time circular confidence gauge, animated verdict shields, SHAP attribution bars, feature telemetry inspector, model performance hub, and session audit history.

---

## Dataset Analysis (Phase 1)

- **Source:** PhiUSIIL Phishing URL Dataset (`data/PhiUSIIL_Phishing_URL_Dataset.csv`)
- **Total Records:** 235,795 rows
- **Total Columns:** 56 columns
- **Target Convention:**
  - `1` = **Legitimate** (134,850 records, 57.19%)
  - `0` = **Phishing** (100,945 records, 42.81%)
- **Class Ratio:** 1.34:1 (Well-balanced)
- **Missing Values:** 0
- **Deduplication:** 425 duplicate URLs removed prior to splitting to eliminate train/test leakage.

---

## Actual Experimental Evaluation Results (Phases 4, 5 & 6)

Evaluated on an unseen test set of **20,000 stratified samples** (80/20 train/test split, random seed = 42):

| Model Architecture | Accuracy | Phishing Recall | Phishing Precision | Phishing F1 | False Negative Rate (Missed) | False Positive Rate (Alarms) | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 99.72% | 99.43% | 99.93% | 0.9968 | 0.57% (49 missed) | 0.05% (6 alarms) | 0.9996 | 0.40s |
| **Random Forest** | 99.66% | 99.37% | 99.84% | 0.9960 | 0.63% (54 missed) | 0.12% (14 alarms) | 0.9996 | 0.56s |
| **XGBoost (Winner)** | **99.78%** | **99.52%** | **99.91%** | **0.9974** | **0.48% (41 missed)** | **0.07% (8 alarms)** | **0.9996** | **0.39s** |

### Winning Model Selection Rationale
In cybersecurity phishing detection, missing an active phishing attack (False Negative) can result in immediate credential theft or ransomware infection. **XGBoost** was selected according to the composite security criterion because it achieved the **highest Phishing Recall (99.52%)**, the **lowest False Negative Rate (0.48%)**, the **highest Phishing F1-Score (0.9974)**, and the **highest overall Accuracy (99.78%)**.

All metrics, confusion matrices, and feature importance charts are saved in the [`results/`](results/) directory.

---

## System Architecture

```text
User Input URL
     │
     ▼
Canonicalization & Normalization (ml/feature_extraction.py)
     │
     ▼
34 URL Feature Extractions (Lexical, Structural, Security, Statistical)
     │
     ▼
Standardized Preprocessing (models/preprocessor.joblib)
     │
     ▼
Trained Best Model: XGBoost (models/best_model.joblib)
     │
     ├──────────────────────────────────────────┐
     ▼                                          ▼
Confidence & Risk Assessment              SHAP Explainability Engine
(models/threshold_config.json)            (ml/explain.py)
     │                                          │
     ▼                                          ▼
Adaptive Supervisory RL Agent             Feature Impact Attributions &
(ml/rl_agent.py)                          Human-Readable Reasoning
     │                                          │
     └─────────────────────┬────────────────────┘
                           ▼
              FastAPI Endpoint (POST /predict)
                           ▼
          Cybersecurity SOC Dashboard (frontend/)
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+ installed
- OpenMP runtime for macOS: `brew install libomp` (already configured)

### 1. Clone & Set Up Virtual Environment
```bash
cd /Users/christinajoseph/Documents/GuardURL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Comprehensive Test Suite
```bash
PYTHONPATH=. ./venv/bin/python tests/test_pipeline.py
```

### 3. Re-Train Models (Optional)
To retrain and evaluate all three models from scratch on 100,000 samples:
```bash
PYTHONPATH=. ./venv/bin/python ml/train.py --samples 100000
```

### 4. Start the Application Server
```bash
PYTHONPATH=. ./venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Then open your browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## API Documentation

### 1. URL Prediction: `POST /predict`
**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Response:**
```json
{
  "url": "https://example.com",
  "prediction": "LEGITIMATE",
  "confidence": 0.9983,
  "confidence_percentage": "99.8%",
  "confidence_level": "HIGH",
  "risk_level": "LOW",
  "probabilities": {
    "legitimate": 0.9983,
    "phishing": 0.0017
  },
  "model_used": "XGBoost",
  "important_factors": [
    "Secure HTTPS encryption enabled",
    "Normal domain length (11 chars)",
    "Normal dot count",
    "Clean alphanumeric domain pattern",
    "Normal hyphen usage"
  ],
  "feature_explanations": [
    {
      "feature": "is_https",
      "value": 1.0,
      "shap_value": 2.0195,
      "abs_impact": 2.0195,
      "pushes_toward": "LEGITIMATE",
      "explanation": "Secure HTTPS encryption enabled"
    }
  ],
  "adaptive_rl_assessment": {
    "action": "ACCEPT_INITIAL_PREDICTION",
    "confidence_tier": "HIGH",
    "threat_flags_detected": false,
    "rationale": "High statistical confidence (99.8%) with congruent feature alignment. The classifier decision is finalized directly."
  },
  "latency_ms": 4.54,
  "disclaimer": "This is an AI/ML probabilistic assessment and not an absolute security guarantee..."
}
```

### 2. Model Metrics: `GET /metrics`
Returns benchmark evaluation metrics, confusion matrices, and training times across all evaluated models.

### 3. Health Check: `GET /health`
Returns system status, active model type, and operational timestamp.

---

## Project Structure

```text
GuardURL/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv      # Ground truth dataset
│
├── ml/
│   ├── __init__.py
│   ├── data_loader.py                         # Deduplication & dataset preparation
│   ├── feature_extraction.py                  # 34-feature extraction engine & normalizer
│   ├── preprocessing.py                       # Preprocessor pipeline (StandardScaler)
│   ├── train.py                               # Model training & benchmark evaluation
│   ├── evaluate.py                            # Standalone evaluation module
│   ├── predict.py                             # Inference engine & risk calculator
│   ├── explain.py                             # SHAP TreeExplainer & XAI reasoner
│   └── rl_agent.py                            # Decoupled Adaptive RL supervisory layer
│
├── models/
│   ├── best_model.joblib                      # Winning model (XGBoost)
│   ├── preprocessor.joblib                    # Fitted scaler
│   ├── feature_names.json                     # Feature list
│   ├── metadata.json                          # Training metadata & benchmark logs
│   ├── threshold_config.json                  # Configurable confidence boundaries
│   └── tld_prob_map.json                      # Empirical TLD reputation lookup
│
├── backend/
│   ├── __init__.py
│   └── app.py                                 # FastAPI application & static mount
│
├── frontend/
│   ├── index.html                             # Modern cybersecurity dashboard
│   ├── style.css                              # Glassmorphic dark cyber styles
│   └── script.js                              # Gauge animations, SHAP bars, API client
│
├── results/
│   ├── dataset_analysis_report.md             # Dataset exploration findings
│   ├── dataset_inspection.json                # Initial dataset telemetry
│   ├── evaluation_summary.md                  # Benchmark summary
│   ├── metrics_comparison.json                # Real model metrics
│   ├── confusion_matrix_all.png               # Confusion matrix visualizations
│   ├── model_comparison.png                   # Performance bar chart
│   └── feature_importance.png                 # Top predictive features chart
│
├── tests/
│   └── test_pipeline.py                       # 10-point automated test suite
│
├── requirements.txt                           # Python dependencies
├── README.md                                  # Complete project documentation
└── .gitignore                                 # Git exclusions
```

---

## Limitations & Operational Scope

1. **Lexical and Structural Scope:** GuardURL performs instantaneous, safe lexical and structural URL analysis without downloading executable payloads or executing malicious JavaScript, avoiding unsafe exposure.
2. **Dynamic Zero-Day Cloaking:** Brand-new compromised legitimate domains (e.g., legitimate high-reputation WordPress sites that have been hacked) may exhibit legitimate structural traits while hosting malicious forms. For critical corporate environments, GuardURL should be deployed alongside real-time domain age and DNS threat intelligence feeds.
3. **Adaptive RL Component:** The Reinforcement Learning component functions as a supervisory decision policy layer above the base classifier. In production, this layer is designed to log ambiguous samples for analyst triage and periodically update its Q-table from analyst feedback.
