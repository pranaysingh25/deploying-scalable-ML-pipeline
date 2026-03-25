"""
Inference Module

This module handles model inference for income prediction.
It loads pre-trained models and preprocessors (encoder, label binarizer) and
provides a predict function that:
1. Converts Pydantic input to appropriate format
2. Processes features using the trained encoder
3. Generates predictions using the trained model
4. Converts model output to human-readable labels

The model and preprocessors are loaded once at module import time for efficiency.
"""

import pickle
import pandas as pd
from pathlib import Path
from starter.ml.data import process_data
from starter.ml.model import inference

# Load model and preprocessors once at module import time for efficiency
model_dir = Path(__file__).parent.parent.parent / "model"

with open(model_dir / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(model_dir / "encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open(model_dir / "lb.pkl", "rb") as f:
    lb = pickle.load(f)

def predict(data):
    """
    Make a prediction using the loaded model.
    
    Transforms input census data using the pre-trained encoder and generates
    an income prediction using the trained Random Forest model.
    
    Parameters
    ----------
    data : CensusData
        Pydantic model containing census features
    
    Returns
    -------
    str
        Prediction label: ">50K" if income exceeds $50K, "<=50K" otherwise
    """
    # Convert Pydantic model to dict with aliases
    input_dict = data.model_dump(by_alias=True)
    
    # Create DataFrame
    df = pd.DataFrame([input_dict])
    
    # Define categorical features
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    
    # Process data using the trained encoder
    X_processed, _, _, _ = process_data(
        df,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Make prediction
    prediction = inference(model, X_processed)[0]
    
    # Convert to label
    predicted_label = ">50K" if prediction == 1 else "<=50K"
    
    return predicted_label

