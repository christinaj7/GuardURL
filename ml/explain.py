"""
GuardURL - Phase 8: Explainable AI (XAI) with SHAP
Generates feature attribution and human-understandable explanations for any URL prediction.
"""

import os
import joblib
import numpy as np
import shap

from ml.feature_extraction import FEATURE_NAMES


class ModelExplainer:
    """
    Computes local SHAP (SHapley Additive exPlanations) values
    and translates them into human-readable cybersecurity reasoning.
    """

    def __init__(self, model=None, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes TreeExplainer for tree models or Explainer fallback."""
        if self.model is None:
            model_path = "models/best_model.joblib"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                return

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback for linear or non-tree models
            self.explainer = shap.Explainer(self.model)

    def explain(self, features_dict: dict, scaled_vector: np.ndarray, top_k: int = 6) -> list[dict]:
        """
        Calculates SHAP contributions for a single prediction vector.
        Returns a list of structured explanations sorted by impact magnitude.
        
        Target convention:
          Positive SHAP (>0) -> pushes toward 1 (LEGITIMATE)
          Negative SHAP (<0) -> pushes toward 0 (PHISHING)
        """
        if self.explainer is None:
            self._init_explainer()

        try:
            shap_vals = self.explainer.shap_values(scaled_vector)
            # Handle multi-output or 2D array
            if isinstance(shap_vals, list):
                # Class 1 (Legitimate) shap values
                raw_values = np.array(shap_vals[1]).flatten() if len(shap_vals) > 1 else np.array(shap_vals[0]).flatten()
            elif len(np.array(shap_vals).shape) == 3:
                # Shape (1, features, classes)
                raw_values = np.array(shap_vals)[0, :, 1]
            else:
                raw_values = np.array(shap_vals).flatten()
        except Exception as e:
            # Fallback heuristic if SHAP fails on edge cases
            raw_values = np.zeros(len(FEATURE_NAMES))

        # Rank features by absolute impact
        feature_impacts = []
        for i, feat_name in enumerate(FEATURE_NAMES):
            val = float(raw_values[i]) if i < len(raw_values) else 0.0
            feature_val = float(features_dict.get(feat_name, 0.0))
            direction = "LEGITIMATE" if val > 0 else "PHISHING"
            feature_impacts.append({
                "feature": feat_name,
                "value": feature_val,
                "shap_value": round(val, 4),
                "abs_impact": abs(val),
                "pushes_toward": direction,
                "explanation": self._get_readable_phrase(feat_name, feature_val, direction)
            })

        # Sort by absolute impact descending
        feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
        return feature_impacts[:top_k]

    def _get_readable_phrase(self, feature: str, value: float, direction: str) -> str:
        """Translates raw feature names and values into cybersecurity analyst explanations."""
        phishing = (direction == "PHISHING")

        if feature == "is_https":
            return "Insecure HTTP protocol (no SSL/TLS encryption)" if phishing else "Secure HTTPS encryption enabled"
        elif feature == "url_length":
            return f"Unusually long URL ({int(value)} characters)" if (phishing and value > 60) else f"Standard URL length ({int(value)} chars)"
        elif feature == "domain_length":
            return f"Abnormally long domain name ({int(value)} chars)" if (phishing and value > 25) else f"Normal domain length ({int(value)} chars)"
        elif feature == "is_ip":
            return "Raw IP address used as hostname instead of a verified domain" if value > 0 else "Standard domain name resolution"
        elif feature == "no_of_subdomains":
            return f"Multiple nested subdomains detected ({int(value)} subdomains)" if (phishing and value >= 2) else "Standard subdomain hierarchy"
        elif feature == "url_entropy":
            return f"High character entropy ({value:.2f}), suggesting obfuscation or random token" if phishing else f"Normal lexical entropy ({value:.2f})"
        elif feature == "has_suspicious_keyword" or feature == "suspicious_keyword_count":
            return f"Phishing lure keyword detected in URL ({int(value)} keyword(s))" if phishing else "No suspicious phishing keywords"
        elif feature == "path_depth":
            return f"Deeply nested directory path ({int(value)} segments)" if (phishing and value > 2) else "Normal path structure"
        elif feature == "digit_ratio" or feature == "no_of_digits":
            return f"High density of digits in URL ({int(value)} digits)" if phishing else "Normal digit distribution"
        elif feature == "letter_ratio":
            return "Low alphabetical letter ratio (heavy symbol/number usage)" if phishing else "High proportion of natural alphabetic text"
        elif feature == "tld_legitimate_prob":
            return f"Top-Level Domain has low historical legitimacy score ({value:.2f})" if phishing else f"Trusted Top-Level Domain reputation ({value:.2f})"
        elif feature == "has_obfuscation" or feature == "no_of_obfuscated_char":
            return f"Percent-encoded or hex obfuscation detected ({int(value)} chars)" if phishing else "No character obfuscation detected"
        elif feature == "no_of_hyphens":
            return f"Excessive hyphens in domain/URL ({int(value)} hyphens)" if (phishing and value > 2) else "Normal hyphen usage"
        elif feature == "no_of_at":
            return "At-symbol (@) detected in URL (used to mask actual host destination)" if value > 0 else "No embedded user credentials"
        elif feature == "no_of_dots":
            return f"High count of dot delimiters ({int(value)} dots)" if (phishing and value > 2) else "Normal dot count"
        elif feature == "tld_in_path":
            return "Top-level domain embedded deceptively inside URL path" if value > 0 else "No deceptive TLD in path"
        elif feature == "has_punycode":
            return "Punycode (xn--) detected; possible homograph spoofing attack" if value > 0 else "Standard ASCII hostname"
        elif feature == "has_port":
            return "Non-standard network port specified in URL" if value > 0 else "Standard default port"
        elif feature == "has_double_slash_redirect":
            return "Double slash redirect (//) detected in path" if value > 0 else "Normal path delimiter"
        elif feature == "char_continuation_rate":
            return "Fragmented domain or lure structure detected" if phishing else "Clean alphanumeric domain pattern"
        elif feature == "url_char_prob":
            return "Irregular character frequency distribution" if phishing else "Natural language character distribution"
        elif feature == "special_char_ratio" or feature == "no_of_special_chars":
            return f"Elevated ratio of special characters ({int(value)} special chars)" if phishing else "Low special character density"
        else:
            return f"{feature.replace('_', ' ').capitalize()} indicates {direction.lower()}"
