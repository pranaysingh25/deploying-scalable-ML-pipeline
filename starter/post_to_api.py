"""
API POST Request Script

This script sends a POST request to the deployed Census Income Prediction API
and returns both the model inference result and the HTTP status code.
"""

import requests


def post_to_api(api_url, census_data):
    """
    Send a POST request to the census prediction API.

    Parameters
    ----------
    api_url : str
        URL of the API endpoint (e.g., https://<domain>/predict)
    census_data : dict
        Dictionary containing census data with hyphenated field names

    Returns
    -------
    dict
        Dictionary containing status_code and prediction result
    """
    response = requests.post(api_url, json=census_data)
    condition = response.status_code == 200
    prediction = response.json().get('prediction') if condition else None
    return {
        'status_code': response.status_code,
        'prediction': prediction
    }


if __name__ == "__main__":
    # Live API endpoint
    api_url = (
        "https://deploying-scalable-ml-pipeline-mkgn.onrender.com/predict"
    )
    API_URL = api_url

    # Sample census data
    census_data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 77516,
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

    # Send POST request
    result = post_to_api(API_URL, census_data)

    # Print results
    print(f"Status Code: {result['status_code']}")
    print(f"Prediction: {result['prediction']}")
