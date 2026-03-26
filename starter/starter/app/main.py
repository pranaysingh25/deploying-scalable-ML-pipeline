"""
FastAPI Application Module

This module implements the REST API for the census income prediction model.
It provides endpoints for:
- Health check / welcome message
- Income prediction based on census data

The API uses Pydantic models for request validation and automatic
documentation generation through FastAPI's Swagger UI.
"""

from fastapi import FastAPI
from .schemas import CensusData
from .inferences import predict

app = FastAPI(
    title="Census Income Prediction API",
    description="Predict whether an individual earns >$50K "
    "based on census data",
    version="1.0.0"
)


@app.get("/")
def read_root() -> dict:
    """
    Root endpoint - returns API welcome message and documentation.

    Returns
    -------
    dict
        Welcome message, description, and link to API documentation
    """
    return {
        "message": "Welcome to the Census Income Prediction API!",
        "description": "Send a POST request to /predict with "
        "census data to get income predictions",
        "docs": "/docs"
    }


@app.post("/predict")
def predict_income(data: CensusData) -> dict:
    """
    Predict income level based on census data.

    Takes census data characteristics as input and predicts whether
    the individual's income exceeds $50K or not using the trained
    Random Forest model.

    Parameters
    ----------
    data : CensusData
        Pydantic model containing all required census features

    Returns
    -------
    dict
        Prediction result with keys:
        - prediction: str, either ">50K" or "<=50K"
    """
    prediction = predict(data)
    return {"prediction": prediction}
