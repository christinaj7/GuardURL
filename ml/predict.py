"""
GuardURL - Phase 7 & 9: URL Prediction Engine
End-to-end inference pipeline supporting single URL evaluation,
risk calculation, confidence tiering, SHAP explainability, and optional RL decision layers.
"""

import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.feature_extraction import extract_url_features, FEATURE_NAMES
from ml.preprocessing import URLPreprocessor
from ml.explain import ModelExplainer
from ml.rl_agent import AdaptiveRLAgent


class GuardURLPredictor:
    """
    Main prediction and inference orchestrator for GuardURL.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.preprocessor = None
        self.threshold_config = {}
        self.tld_prob_map = {}
        self.explainer = None
        self.rl_agent = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads trained model, preprocessor, configuration, and auxiliary components."""
        model_path = os.path.join(self.models_dir, "best_model.joblib")
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.joblib")
        config_path = os.path.join(self.models_dir, "threshold_config.json")
        tld_path = os.path.join(self.models_dir, "tld_prob_map.json")

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Trained model or preprocessor not found. Please train models first using ml/train.py.")

        self.model = joblib.load(model_path)
        self.preprocessor = URLPreprocessor.load(preprocessor_path)

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.threshold_config = json.load(f)
        else:
            self.threshold_config = {
                "classification_threshold": 0.50,
                "high_confidence_phishing_threshold": 0.80,
                "high_confidence_legitimate_threshold": 0.80,
                "uncertain_confidence_lower": 0.35,
                "uncertain_confidence_upper": 0.65
            }

        if os.path.exists(tld_path):
            with open(tld_path, "r") as f:
                self.tld_prob_map = json.load(f)

        self.explainer = ModelExplainer(model=self.model, preprocessor=self.preprocessor)
        self.rl_agent = AdaptiveRLAgent()

    def predict(self, url: str, enable_rl: bool = True) -> dict:
        """
        Executes complete inference on a raw URL.
        Returns prediction, probability, confidence, risk level, SHAP explanations,
        and adaptive decision assessment.
        """
        # 1. Validation
        if not url or not str(url).strip():
            raise ValueError("Input URL cannot be empty.")

        url_str = str(url).strip()

        # 2. Feature Extraction
        features = extract_url_features(url_str, tld_prob_map=self.tld_prob_map)

        # 3. Preprocessing
        X_scaled = self.preprocessor.transform_single(features)

        # 4. Model Prediction
        # Label 1 = Legitimate, Label 0 = Phishing
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
            prob_phishing = float(probs[0])
            prob_legit = float(probs[1])
        else:
            pred_raw = int(self.model.predict(X_scaled)[0])
            prob_legit = 1.0 if pred_raw == 1 else 0.0
            prob_phishing = 1.0 - prob_legit

        threshold = float(self.threshold_config.get("classification_threshold", 0.50))
        is_legitimate = (prob_legit >= threshold)
        prediction = "LEGITIMATE" if is_legitimate else "PHISHING"
        confidence = prob_legit if is_legitimate else prob_phishing

        # 5. Risk Assessment
        if prob_phishing >= 0.85:
            risk_level = "CRITICAL"
        elif prob_phishing >= 0.60:
            risk_level = "HIGH"
        elif prob_phishing >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 6. Confidence Assessment
        if confidence >= 0.80:
            confidence_level = "HIGH"
        elif confidence >= 0.65:
            confidence_level = "MODERATE"
        else:
            confidence_level = "UNCERTAIN"

        # 7. Adaptive / RL Decision Layer (Phase 10)
        rl_info = None
        if enable_rl and self.rl_agent:
            rl_decision = self.rl_agent.select_action(prediction, confidence, features)
            if rl_decision["action"] == "PERFORM_ADDITIONAL_ANALYSIS":
                analysis = self.rl_agent.perform_additional_analysis(features, prob_legit)
                rl_decision["additional_analysis"] = analysis
            rl_info = rl_decision

        # 8. Explainability via SHAP (Phase 8)
        explanations = self.explainer.explain(features, X_scaled, top_k=6)
        important_factors = [exp["explanation"] for exp in explanations]

        # 9. Structured Response
        result = {
            "url": url_str,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "confidence_level": confidence_level,
            "risk_level": risk_level,
            "probabilities": {
                "legitimate": round(prob_legit, 4),
                "phishing": round(prob_phishing, 4)
            },
            "model_used": self.threshold_config.get("model_name", "Best Model"),
            "important_factors": important_factors,
            "feature_explanations": explanations,
            "raw_features": features,
            "adaptive_rl_assessment": rl_info,
            "disclaimer": "This is an AI/ML probabilistic assessment and not an absolute security guarantee. Always verify SSL certificates and domain credentials before entering private data."
        }

        return result


# Global singleton instance for high-throughput backend caching
_predictor_instance = None


def get_predictor() -> GuardURLPredictor:
    """Returns singleton instance of GuardURLPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = GuardURLPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = get_predictor()
    test_urls = [
        "https://www.google.com/search?q=machine+learning",
        "http://paypal.account-verify.security-update.xyz/login?session=928374@auth",
        "http://192.168.1.100/admin/bank-login.html"
    ]

    for u in test_urls:
        res = predictor.predict(u)
        print("\n" + "=" * 60)
        print(f"URL: {res['url']}")
        print(f"Prediction: {res['prediction']} ({res['confidence_percentage']}) | Risk: {res['risk_level']}")
        print("Important Factors:")
        for factor in res["important_factors"][:4]:
            print(f"  • {factor}")
        if res["adaptive_rl_assessment"]:
            print(f"RL Action: {res['adaptive_rl_assessment']['action']} ({res['adaptive_rl_assessment']['rationale'][:60]}...)")
