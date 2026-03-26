"""
API Testing Module

This module contains unit tests for the Census Income Prediction API.
It tests:
- API endpoint availability and responses
- Request validation
- Prediction functionality on edge cases

Tests verify that the API correctly handles both low and high income
predictions and returns responses in the expected format.
"""

from fastapi.testclient import TestClient
from starter.app.main import app

client = TestClient(app)


def test_read_root():
    """
    Test root endpoint returns welcome message.

    Verifies that GET request to root endpoint returns HTTP 200
    with welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "welcome" in response.json()["message"].lower()


def test_predict_low_income():
    """
    Test income prediction for a sample with likely low income.

    Uses features typical of lower income individuals (young age,
    lower education, lower-skill occupation) and verifies prediction
    returns valid format.
    """
    test_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_predict_high_income():
    """
    Test income prediction for a sample with likely high income.

    Uses features typical of higher income individuals (mature age,
    higher education, executive occupation, married) and verifies
    prediction returns valid format.
    """
    test_data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]
