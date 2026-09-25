"""
GuardURL - Phase 4 & 5: Training and Evaluation Pipeline
Trains Logistic Regression, Random Forest, and XGBoost.
Evaluates using genuine metrics, generates comparison visualizations,
and selects the best performing model based on documented cybersecurity criteria.
"""

import os
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.data_loader import prepare_dataset
from ml.preprocessing import URLPreprocessor
from ml.feature_extraction import FEATURE_NAMES

SEED = 42
RESULTS_DIR = "results"
MODELS_DIR = "models"


def compute_metrics(y_true, y_pred, y_prob=None, model_name="Model") -> dict:
    """
    Computes rigorous cybersecurity evaluation metrics.
    Note on labels:
      y = 1 -> Legitimate
      y = 0 -> Phishing (Threat class)
    """
    acc = accuracy_score(y_true, y_pred)
    # Phishing (label 0) is the primary threat class
    prec_phish = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_phish = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_phish = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Legitimate (label 1)
    prec_legit = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_legit = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_legit = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Macro & Weighted F1
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # Confusion matrix with explicit labels [0, 1]
    # cm[0,0] = Actual 0, Pred 0 (True Phishing)
    # cm[0,1] = Actual 0, Pred 1 (False Legitimate / False Negative Phishing)
    # cm[1,0] = Actual 1, Pred 0 (False Phishing / False Positive Phishing)
    # cm[1,1] = Actual 1, Pred 1 (True Legitimate)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp_phish = int(cm[0, 0])
    fn_phish = int(cm[0, 1])  # Phishing missed
    fp_phish = int(cm[1, 0])  # Legitimate blocked
    tn_phish = int(cm[1, 1])

    # Cybersecurity threat detection rates:
    # FNR on Phishing = Phishing missed / Total Phishing
    fnr_phish = fn_phish / max((fn_phish + tp_phish), 1)
    # FPR on Phishing = Legitimate wrongly flagged / Total Legitimate
    fpr_phish = fp_phish / max((fp_phish + tn_phish), 1)

    auc_score = 0.0
    if y_prob is not None:
        try:
            # probability of legitimate class (1)
            auc_score = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc_score = 0.0

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc": round(float(auc_score), 4),
        "phishing_metrics": {
            "precision": round(float(prec_phish), 4),
            "recall": round(float(rec_phish), 4),
            "f1_score": round(float(f1_phish), 4),
            "false_negative_rate": round(float(fnr_phish), 4),
            "false_positive_rate": round(float(fpr_phish), 4),
            "true_positives": tp_phish,
            "false_negatives": fn_phish,
            "false_positives": fp_phish,
            "true_negatives": tn_phish
        },
        "legitimate_metrics": {
            "precision": round(float(prec_legit), 4),
            "recall": round(float(rec_legit), 4),
            "f1_score": round(float(f1_legit), 4)
        },
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrices(metrics_dict: dict, output_path: str):
    """Plots confusion matrices for all evaluated models side-by-side."""
    models = list(metrics_dict.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        cm = np.array(metrics_dict[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Phishing (0)", "Pred Legit (1)"],
            yticklabels=["Actual Phishing (0)", "Actual Legit (1)"]
        )
        ax.set_title(f"{name}\nAcc: {metrics_dict[name]['accuracy']:.3f} | F1: {metrics_dict[name]['f1_macro']:.3f}")
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Model Prediction")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrices visualization to {output_path}")


def plot_model_comparison(metrics_dict: dict, output_path: str):
    """Bar chart comparing key performance metrics across all models."""
    models = list(metrics_dict.keys())
    metrics_names = ["Accuracy", "Phishing Recall", "Phishing F1", "Macro F1", "ROC-AUC"]

    data = {m: [
        metrics_dict[m]["accuracy"],
        metrics_dict[m]["phishing_metrics"]["recall"],
        metrics_dict[m]["phishing_metrics"]["f1_score"],
        metrics_dict[m]["f1_macro"],
        metrics_dict[m]["roc_auc"]
    ] for m in models}

    x = np.arange(len(metrics_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4A90E2", "#50E3C2", "#B8E986"]

    for i, m in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, data[m], width, label=m, color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("GuardURL — Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved model comparison visualization to {output_path}")


def plot_feature_importance(feature_names: list[str], importances: list[float], output_path: str, title: str):
    """Plots top 15 feature importances for tree-based models."""
    indices = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(top_features)), top_importances[::-1], color="#38bdf8", edgecolor="#0284c7")
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=10)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved feature importance visualization to {output_path}")


def train_and_evaluate(sample_size: int = 100000):
    """
    Executes Phase 4 & Phase 5:
    1. Loads dataset and extracts URL features
    2. Performs stratified train/test split (80/20)
    3. Fits preprocessor ONLY on train split
    4. Trains Logistic Regression, Random Forest, XGBoost
    5. Evaluates genuine test metrics
    6. Generates visual plots and saves JSON results
    7. Selects best model according to documented criteria
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("==================================================")
    print("PHASE 4: MACHINE LEARNING MODEL TRAINING")
    print("==================================================")
    print(f"Random Seed: {SEED} | Configured Sample Size: {sample_size or 'Full Dataset'}")

    # 1. Prepare data
    X, y, tld_prob_map = prepare_dataset(sample_size=sample_size)
    print(f"Loaded dataset: X={X.shape}, y distribution: {dict(y.value_counts())}")

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"Train split: {X_train.shape[0]:,} samples | Test split: {X_test.shape[0]:,} samples")

    # 3. Fit Preprocessor ONLY on training data
    print("Fitting URLPreprocessor on training data...")
    preprocessor = URLPreprocessor(FEATURE_NAMES)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    preprocessor.save(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    # Also save raw feature names
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    # 4. Define and train models
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=SEED,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=5,
            n_jobs=-1,
            random_state=SEED,
            class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=SEED,
            eval_metric="logloss"
        )
    }

    metrics_results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        # For tree models, unscaled or scaled can be used; scaled is consistent
        model.fit(X_train_scaled, y_train)
        train_time = round(time.time() - t0, 2)
        print(f"Completed training {name} in {train_time}s.")

        # Predictions on unseen test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        res = compute_metrics(y_test, y_pred, y_prob, model_name=name)
        res["train_time_seconds"] = train_time
        metrics_results[name] = res
        trained_models[name] = model

        print(f"[{name}] Test Accuracy: {res['accuracy']:.4f} | Phishing F1: {res['phishing_metrics']['f1_score']:.4f} | Phishing Recall: {res['phishing_metrics']['recall']:.4f} | FNR: {res['phishing_metrics']['false_negative_rate']:.4f}")

    # 5. Save metrics to JSON
    metrics_path = os.path.join(RESULTS_DIR, "metrics_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"\nSaved evaluation metrics to {metrics_path}")

    # 6. Generate Visualizations
    cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrix_all.png")
    plot_confusion_matrices(metrics_results, cm_plot_path)

    comp_plot_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plot_model_comparison(metrics_results, comp_plot_path)

    # Feature importance for tree models
    if "XGBoost" in trained_models and hasattr(trained_models["XGBoost"], "feature_importances_"):
        fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
        plot_feature_importance(
            FEATURE_NAMES,
            trained_models["XGBoost"].feature_importances_,
            fi_path,
            "GuardURL — XGBoost Top Predictive Features"
        )
    elif "Random Forest" in trained_models and hasattr(trained_models["Random Forest"], "feature_importances_"):
        fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
        plot_feature_importance(
            FEATURE_NAMES,
            trained_models["Random Forest"].feature_importances_,
            fi_path,
            "GuardURL — Random Forest Top Predictive Features"
        )

    # ==================================================
    # PHASE 6: MODEL SELECTION
    # ==================================================
    print("\n==================================================")
    print("PHASE 6: MODEL SELECTION & ARTIFACT EXPORT")
    print("==================================================")
    # Selection criteria:
    # In cybersecurity phishing detection:
    # 1. Phishing Recall & Low False Negative Rate (FNR): Missing a phishing site is catastrophic (credential theft/ransomware).
    # 2. Phishing F1-Score: Harmonic balance between phishing precision and recall.
    # 3. Overall Accuracy and Macro F1.
    best_model_name = None
    best_score = -1.0

    print("Evaluating selection criteria:")
    for name, res in metrics_results.items():
        # Weighted composite score: 40% Phishing F1 + 30% Phishing Recall + 30% Overall Accuracy
        score = (
            0.40 * res["phishing_metrics"]["f1_score"] +
            0.30 * res["phishing_metrics"]["recall"] +
            0.30 * res["accuracy"]
        )
        print(f"  {name}: Composite Security Score = {score:.4f} (Phishing Recall={res['phishing_metrics']['recall']:.4f}, Phishing F1={res['phishing_metrics']['f1_score']:.4f}, Acc={res['accuracy']:.4f})")
        if score > best_score:
            best_score = score
            best_model_name = name

    print(f"\nWinning Model Selected: '{best_model_name}' (Composite Score: {best_score:.4f})")

    best_model = trained_models[best_model_name]
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Saved winning model to {best_model_path}")

    # Threshold configuration (Phase 9)
    threshold_config = {
        "model_name": best_model_name,
        "classification_threshold": 0.50,
        "high_confidence_phishing_threshold": 0.80,
        "high_confidence_legitimate_threshold": 0.80,
        "uncertain_confidence_lower": 0.35,
        "uncertain_confidence_upper": 0.65,
        "risk_levels": {
            "CRITICAL": {"min_phishing_prob": 0.85, "badge": "CRITICAL RISK", "color": "#ef4444"},
            "HIGH": {"min_phishing_prob": 0.65, "badge": "HIGH RISK", "color": "#f97316"},
            "MEDIUM": {"min_phishing_prob": 0.40, "badge": "MODERATE / UNCERTAIN", "color": "#eab308"},
            "LOW": {"min_phishing_prob": 0.00, "badge": "LOW RISK", "color": "#22c55e"}
        },
        "disclaimer": "This is an AI/ML-based probabilistic classification and not an absolute security guarantee. Always verify certificates and domains before entering sensitive credentials."
    }
    with open(os.path.join(MODELS_DIR, "threshold_config.json"), "w") as f:
        json.dump(threshold_config, f, indent=2)

    # Metadata
    metadata = {
        "model_name": best_model_name,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": SEED,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_features": len(FEATURE_NAMES),
        "best_model_metrics": metrics_results[best_model_name],
        "all_models_metrics": metrics_results,
        "selection_rationale": "Selected based on composite cybersecurity utility (harmonic balance of phishing detection recall, phishing F1-score, and false negative rate minimization)."
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Generate Markdown Summary Report
    summary_md = f"""# GuardURL — Model Training & Evaluation Summary

## Experimental Configuration
- **Dataset:** PhiUSIIL Phishing URL Dataset
- **Total Samples Evaluated:** {len(X):,}
- **Train Split (80%):** {len(X_train):,} samples
- **Test Split (20%):** {len(X_test):,} unseen samples
- **Feature Count:** {len(FEATURE_NAMES)} URL-based features
- **Random Seed:** {SEED} (Fully Reproducible)

## Verified Evaluation Results on Unseen Test Split

| Model | Accuracy | Phishing Recall | Phishing F1 | Phishing FNR | Macro F1 | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for name, res in metrics_results.items():
        summary_md += f"| **{name}** | {res['accuracy']:.4f} | {res['phishing_metrics']['recall']:.4f} | {res['phishing_metrics']['f1_score']:.4f} | {res['phishing_metrics']['false_negative_rate']:.4f} | {res['f1_macro']:.4f} | {res['roc_auc']:.4f} | {res['train_time_seconds']}s |\n"

    summary_md += f"""
## Model Selection Decision
- **Winning Model:** **{best_model_name}**
- **Selection Criterion:** Maximization of Phishing Detection Recall (minimizing False Negative Rate) combined with Phishing F1-score and overall accuracy.
- **Saved Model Location:** `{best_model_path}`
- **Preprocessor Location:** `{os.path.join(MODELS_DIR, "preprocessor.joblib")}`
"""
    with open(os.path.join(RESULTS_DIR, "evaluation_summary.md"), "w") as f:
        f.write(summary_md)

    print("\nTraining and evaluation pipeline completed successfully!")
    return metrics_results, best_model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GuardURL models.")
    parser.add_argument("--samples", type=int, default=100000, help="Number of samples to train on (default 100000)")
    args = parser.parse_args()
    train_and_evaluate(sample_size=args.samples)
