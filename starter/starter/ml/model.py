"""
Model Training and Inference Module

This module provides functions for training machine learning models and generating predictions.
It implements:
- Model training using Random Forest Classifier
- Model inference on new data
- Performance metric computation (precision, recall, F-beta score)
- Categorical slice analysis for fairness assessment

Functions:
    train_model: Train a Random Forest classifier
    compute_model_metrics: Calculate precision, recall, and F-beta metrics
    inference: Generate predictions using a trained model
    performance_on_categorical_slices: Analyze performance across categorical feature values
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def performance_on_categorical_slices(data, y, preds, categorical_features):
    """
    Compute model performance on slices of categorical features.

    Inputs
    ------
    data : pd.DataFrame
        Original dataframe with categorical features
    y : np.ndarray
        True labels
    preds : np.ndarray
        Predicted labels
    categorical_features : list
        List of categorical feature names

    Returns
    -------
    slice_metrics : dict
        Dictionary containing performance metrics for each categorical feature slice
    """
    slice_metrics = {}
    
    for feature in categorical_features:
        feature_values = data[feature].unique()
        slice_metrics[feature] = {}
        
        for value in feature_values:
            # Get indices where feature equals value
            indices = data[feature] == value
            
            if indices.sum() > 0:  # Only compute if there are samples
                y_slice = y[indices]
                preds_slice = preds[indices]
                
                precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
                
                slice_metrics[feature][value] = {
                    'precision': precision,
                    'recall': recall,
                    'fbeta': fbeta,
                    'count': indices.sum()
                }
    
    return slice_metrics
