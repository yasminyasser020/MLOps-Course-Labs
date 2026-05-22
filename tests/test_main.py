"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

from litestar.testing import TestClient
from main import app
from app.model_utils import predict_churn

# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------


# TODO 1: Write a test that calls predict_churn() directly with sample features
#         and asserts the result is 0 or 1
#         Hint: import predict_churn from app.model_utils
def test_predict_churn_direct_function():
    """Calls the inference function directly with standard sample features."""
    standard_sample = [
        650.0,
        "France",
        "Male",
        35.0,
        3.0,
        50000.0,
        2.0,
        1.0,
        1.0,
        85000.0,
    ]
    result = predict_churn(standard_sample)

    assert isinstance(result, int)
    assert result in [0, 1]


# TODO 2 (bonus): Write another function test with edge-case inputs
def test_predict_churn_edge_cases():
    """Calls the inference function directly with extreme boundary edge cases."""
    # Testing a customer with absolute zero parameters across all numeric metrics
    extreme_zero_sample = [0.0, "Spain", "Female", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    result_zero = predict_churn(extreme_zero_sample)
    assert result_zero in [0, 1]

    # Testing a customer with exceptionally high credit and financial metrics
    whale_sample = [
        850.0,
        "Germany",
        "Male",
        99.0,
        10.0,
        250000.0,
        4.0,
        1.0,
        1.0,
        500000.0,
    ]
    result_whale = predict_churn(whale_sample)
    assert result_whale in [0, 1]


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------

# TODO 3: Write a test that POSTs to /predict with valid JSON
#         and checks the status code and response body
#         Hint: Litestar POST returns 201, not 200
#         Hint: use `with TestClient(app=app) as client:`


def test_post_predict_valid_json():
    """Verifies that sending a correct payload returns a 201 status and a valid binary choice."""
    valid_payload = {
        "CreditScore": 612.0,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42.0,
        "Tenure": 2.0,
        "Balance": 0.0,
        "NumOfProducts": 1.0,
        "HasCrCard": 1.0,
        "IsActiveMember": 1.0,
        "EstimatedSalary": 101348.88,
    }
    with TestClient(app=app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 201

        response_json = response.json()
        assert "prediction" in response_json
        assert response_json["prediction"] in [0, 1]


# TODO 4: Write a test for GET /health


def test_get_health_endpoint():
    """Verifies that the server check returns a 200 and healthy metadata."""
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# TODO 5: Write a test for GET /


def test_get_home_endpoint():
    """Verifies that the landing page returns a 200 and structural greeting message."""
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


# TODO 6 (bonus): Test that invalid input returns status 400
def test_post_predict_invalid_input_returns_400():
    """Verifies that malformed payloads are safely rejected by the Pydantic tier with an HTTP 400."""
    malformed_payload = {
        "CreditScore": "poor-score",  # Type error
        "Geography": "Germany",
        "Gender": "Female",
        "Age": -5.0,  # Constraint error
        "Tenure": 3.0,
        "Balance": 45000.0,  # Missing fields follow...
    }
    with TestClient(app=app) as client:
        response = client.post("/predict", json=malformed_payload)

        # ─── ADD THESE LINES TO VERIFY ─────────────────────────────────────
        print("\n=== LOOK HERE: PYDANTIC ERROR BUNDLE ===")
        import json

        print(json.dumps(response.json(), indent=2))
        print("========================================\n")
        # ───────────────────────────────────────────────────────────────────

        assert response.status_code == 400
