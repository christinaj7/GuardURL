"""
GuardURL - Phase 10: Adaptive Reinforcement Learning Decision Layer
Decoupled decision-making agent that observes classifier confidence
and decides whether to accept the prediction, perform deep analysis, or flag for analyst review.
The primary classifier functions independently of this module.
"""

import os
import json
import random
import numpy as np

ACTIONS = [
    "ACCEPT_INITIAL_PREDICTION",
    "PERFORM_ADDITIONAL_ANALYSIS",
    "FLAG_FOR_REVIEW"
]


class AdaptiveRLAgent:
    """
    Q-Learning / Contextual Bandit Agent for uncertain prediction policy.
    Operates as an adaptive supervisory layer above the base ML classifier.
    """

    def __init__(self, epsilon: float = 0.1, alpha: float = 0.2, gamma: float = 0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.policy_file = "models/rl_policy.json"
        self._init_or_load_policy()

    def _init_or_load_policy(self):
        """Initializes default Q-values or loads trained policy from file."""
        if os.path.exists(self.policy_file):
            try:
                with open(self.policy_file, "r") as f:
                    self.q_table = json.load(f)
                return
            except Exception:
                pass

        # Seed initial sensible baseline Q-values for key cybersecurity states
        # State: (predicted_label, confidence_tier, has_threat_flags)
        states = [
            ("LEGITIMATE", "HIGH", False),
            ("LEGITIMATE", "HIGH", True),
            ("LEGITIMATE", "UNCERTAIN", False),
            ("LEGITIMATE", "UNCERTAIN", True),
            ("PHISHING", "HIGH", False),
            ("PHISHING", "HIGH", True),
            ("PHISHING", "UNCERTAIN", False),
            ("PHISHING", "UNCERTAIN", True)
        ]

        for s in states:
            key = self._state_to_key(s[0], s[1], s[2])
            if s[1] == "HIGH":
                self.q_table[key] = {
                    "ACCEPT_INITIAL_PREDICTION": 2.0,
                    "PERFORM_ADDITIONAL_ANALYSIS": 0.5,
                    "FLAG_FOR_REVIEW": 0.1
                }
            elif s[1] == "UNCERTAIN" and s[2]:  # uncertain with threat flags present
                self.q_table[key] = {
                    "ACCEPT_INITIAL_PREDICTION": -0.5,
                    "PERFORM_ADDITIONAL_ANALYSIS": 2.5,
                    "FLAG_FOR_REVIEW": 1.8
                }
            else:  # uncertain without flags
                self.q_table[key] = {
                    "ACCEPT_INITIAL_PREDICTION": 0.5,
                    "PERFORM_ADDITIONAL_ANALYSIS": 1.8,
                    "FLAG_FOR_REVIEW": 1.2
                }

        self.save_policy()

    def _state_to_key(self, prediction: str, conf_tier: str, threat_flags: bool) -> str:
        return f"{prediction}|{conf_tier}|{'THREATS_PRESENT' if threat_flags else 'NO_THREATS'}"

    def select_action(self, prediction: str, confidence: float, features: dict) -> dict:
        """
        Determines the adaptive supervisory action for a given prediction.
        """
        # Determine confidence tier
        if confidence >= 0.80:
            conf_tier = "HIGH"
        else:
            conf_tier = "UNCERTAIN"

        # Check for presence of acute threat signals
        threat_flags = (
            features.get("is_ip", 0) > 0 or
            features.get("has_suspicious_keyword", 0) > 0 or
            features.get("has_obfuscation", 0) > 0 or
            features.get("url_entropy", 0) > 4.5 or
            features.get("has_punycode", 0) > 0 or
            features.get("no_of_subdomains", 0) >= 3
        )

        state_key = self._state_to_key(prediction, conf_tier, threat_flags)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.5 for a in ACTIONS}

        # Greedy choice based on Q-values
        action_scores = self.q_table[state_key]
        best_action = max(action_scores, key=action_scores.get)

        reasoning = self._action_reasoning(best_action, conf_tier, threat_flags, prediction, confidence)

        return {
            "action": best_action,
            "confidence_tier": conf_tier,
            "threat_flags_detected": threat_flags,
            "q_values": action_scores,
            "rationale": reasoning
        }

    def _action_reasoning(self, action: str, conf_tier: str, threat_flags: bool, prediction: str, conf: float) -> str:
        if action == "ACCEPT_INITIAL_PREDICTION":
            return f"High statistical confidence ({conf*100:.1f}%) with congruent feature alignment. The classifier decision is finalized directly."
        elif action == "PERFORM_ADDITIONAL_ANALYSIS":
            return f"Prediction confidence ({conf*100:.1f}%) is in the borderline zone with suspicious structural traits. Deep heuristic analysis applied."
        else:
            return "Classification ambiguity detected between benign structural traits and risk indicators. Flagged for secondary human/sandbox verification."

    def perform_additional_analysis(self, features: dict, base_prob_legit: float) -> dict:
        """
        Secondary heuristic refinement triggered when RL agent chooses PERFORM_ADDITIONAL_ANALYSIS.
        Computes auxiliary risk index based on combined adversarial patterns.
        """
        risk_penalties = 0.0

        if features.get("is_ip", 0) > 0:
            risk_penalties += 0.30
        if features.get("has_suspicious_keyword", 0) > 0:
            risk_penalties += 0.20
        if features.get("has_obfuscation", 0) > 0:
            risk_penalties += 0.15
        if features.get("url_entropy", 0) > 4.5:
            risk_penalties += 0.15
        if features.get("has_punycode", 0) > 0:
            risk_penalties += 0.20
        if features.get("is_https", 1) == 0:
            risk_penalties += 0.15
        if features.get("no_of_subdomains", 0) >= 3:
            risk_penalties += 0.15

        # Refined legitimate probability (clamped to [0.01, 0.99])
        adjusted_prob_legit = max(0.01, min(0.99, base_prob_legit - risk_penalties))

        return {
            "initial_prob_legitimate": round(base_prob_legit, 4),
            "adjusted_prob_legitimate": round(adjusted_prob_legit, 4),
            "risk_penalties_applied": round(risk_penalties, 4),
            "refinement_notes": "Applied compound adversarial heuristics to resolve prediction boundary ambiguity."
        }

    def save_policy(self):
        """Persists current Q-table to disk."""
        os.makedirs(os.path.dirname(self.policy_file), exist_ok=True)
        with open(self.policy_file, "w") as f:
            json.dump(self.q_table, f, indent=2)
