"""
GuardURL - Phase 14: Comprehensive Verification & Test Suite
Tests training artifacts, model loading, feature extraction, predictions,
invalid inputs, SHAP explanations, Adaptive RL layer, and FastAPI endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.feature_extraction import extract_url_features, FEATURE_NAMES
from ml.preprocessing import URLPreprocessor
from ml.predict import get_predictor
from backend.app import app


client = TestClient(app)


def test_01_feature_names_and_extraction():
    """Verifies that all 34 features are extracted correctly from a URL."""
    url = "https://www.example.com/test?param=1"
    features = extract_url_features(url)

    assert isinstance(features, dict)
    assert len(features) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in features, f"Missing feature: {name}"
        assert isinstance(features[name], (int, float))

    assert features["is_https"] == 1.0
    assert features["is_ip"] == 0.0
    print("Test 1 Passed: Feature extraction completed with 34 features.")


def test_02_model_and_preprocessor_loading():
    """Verifies that model and preprocessor load cleanly from disk."""
    predictor = get_predictor()
    assert predictor.model is not None, "Model failed to load."
    assert predictor.preprocessor is not None, "Preprocessor failed to load."
    assert predictor.explainer is not None, "SHAP explainer failed to initialize."
    assert predictor.rl_agent is not None, "Adaptive RL agent failed to initialize."
    print("Test 2 Passed: Model artifacts loaded cleanly.")


def test_03_prediction_normal_url():
    """Verifies inference on a benign normal URL."""
    predictor = get_predictor()
    url = "https://example.com"
    result = predictor.predict(url)

    assert result["prediction"] in ["LEGITIMATE", "PHISHING"]
    assert result["prediction"] == "LEGITIMATE", f"Expected LEGITIMATE for {url}, got {result['prediction']}"
    assert result["risk_level"] in ["LOW", "MEDIUM"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["important_factors"]) > 0
    assert "disclaimer" in result
    print(f"Test 3 Passed: Normal URL predicted as {result['prediction']} ({result['confidence_percentage']}).")


def test_04_prediction_phishing_url():
    """Verifies inference on a known phishing structure."""
    predictor = get_predictor()
    url = "http://paypal.verify-account.security-update.xyz/signin?token=928374@auth"
    result = predictor.predict(url)

    assert result["prediction"] == "PHISHING", f"Expected PHISHING for {url}, got {result['prediction']}"
    assert result["risk_level"] in ["HIGH", "CRITICAL"]
    assert result["confidence"] > 0.80
    assert len(result["important_factors"]) > 0
    print(f"Test 4 Passed: Phishing URL predicted as {result['prediction']} ({result['confidence_percentage']}).")


def test_05_empty_input_handling():
    """Verifies graceful handling of empty inputs."""
    predictor = get_predictor()
    with pytest.raises(ValueError):
        predictor.predict("")

    with pytest.raises(ValueError):
        predictor.predict("   ")
    print("Test 5 Passed: Empty input validation handled cleanly.")


def test_06_shap_explanation_structure():
    """Verifies SHAP attributions and human-readable explanation formats."""
    predictor = get_predictor()
    url = "http://192.168.1.100/admin/bank-login.html"
    result = predictor.predict(url)

    assert "feature_explanations" in result
    exps = result["feature_explanations"]
    assert len(exps) > 0

    for exp in exps:
        assert "feature" in exp
        assert "shap_value" in exp
        assert "pushes_toward" in exp
        assert exp["pushes_toward"] in ["LEGITIMATE", "PHISHING"]
        assert "explanation" in exp
        assert len(exp["explanation"]) > 0
    print("Test 6 Passed: SHAP explanations structured accurately.")


def test_07_adaptive_rl_assessment():
    """Verifies that the Adaptive RL supervisory agent evaluates predictions."""
    predictor = get_predictor()
    url = "http://paypal.verify-account.security-update.xyz/signin?token=928374@auth"
    result = predictor.predict(url, enable_rl=True)

    rl = result["adaptive_rl_assessment"]
    assert rl is not None
    assert "action" in rl
    assert rl["action"] in ["ACCEPT_INITIAL_PREDICTION", "PERFORM_ADDITIONAL_ANALYSIS", "FLAG_FOR_REVIEW"]
    assert "confidence_tier" in rl
    assert "rationale" in rl
    print(f"Test 7 Passed: Adaptive RL evaluated with action '{rl['action']}'.")


def test_08_api_health_endpoint():
    """Verifies GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    print("Test 8 Passed: API /health endpoint active.")


def test_09_api_predict_endpoint_valid_and_invalid():
    """Verifies POST /predict endpoint with both valid and invalid requests."""
    # Valid call
    resp_valid = client.post("/predict", json={"url": "https://example.com"})
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert data["prediction"] == "LEGITIMATE"
    assert "latency_ms" in data

    # Empty URL error
    resp_empty = client.post("/predict", json={"url": ""})
    assert resp_empty.status_code == 400

    # Missing field error
    resp_missing = client.post("/predict", json={})
    assert resp_missing.status_code == 422
    print("Test 9 Passed: API /predict endpoint tested for valid & invalid payloads.")


def test_10_api_metrics_endpoint():
    """Verifies GET /metrics endpoint returns genuine trained metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models_comparison" in data
    assert "XGBoost" in data["models_comparison"]
    assert "Random Forest" in data["models_comparison"]
    assert "Logistic Regression" in data["models_comparison"]
    print("Test 10 Passed: API /metrics endpoint returned all 3 trained models.")


if __name__ == "__main__":
    test_01_feature_names_and_extraction()
    test_02_model_and_preprocessor_loading()
    test_03_prediction_normal_url()
    test_04_prediction_phishing_url()
    test_05_empty_input_handling()
    test_06_shap_explanation_structure()
    test_07_adaptive_rl_assessment()
    test_08_api_health_endpoint()
    test_09_api_predict_endpoint_valid_and_invalid()
    test_10_api_metrics_endpoint()
    print("\n==============================================")
    print("ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==============================================")
