"""
Unit tests for the ML model training and evaluation functions.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, performance_on_categorical_slices


class TestDataProcessing:
    """Tests for data processing function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = pd.DataFrame({
            'age': [25, 45, 35, 28, 55],
            'workclass': ['Private', 'Self-emp-not-inc', 'Private', 'Government', 'Private'],
            'education': ['Bachelors', 'Masters', 'HS-grad', 'Bachelors', 'Doctorate'],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Married-civ-spouse'],
            'occupation': ['Tech-support', 'Exec-managerial', 'Craft-repair', 'Adm-clerical', 'Prof-specialty'],
            'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Own-child', 'Husband'],
            'race': ['White', 'White', 'Black', 'White', 'Asian'],
            'sex': ['Male', 'Male', 'Male', 'Male', 'Male'],
            'native-country': ['United-States', 'United-States', 'United-States', 'Cuba', 'United-States'],
            'salary': ['<=50K', '>50K', '<=50K', '<=50K', '>50K']
        })
        return data
    
    def test_process_data_training_mode(self, sample_data):
        """Test process_data function in training mode."""
        cat_features = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
        
        X, y, encoder, lb = process_data(
            sample_data,
            categorical_features=cat_features,
            label='salary',
            training=True
        )
        
        # Check output shapes
        assert X.shape[0] == len(sample_data)
        assert y.shape[0] == len(sample_data)
        assert encoder is not None
        assert lb is not None
        
        # Check that labels are binary (0 or 1)
        assert set(y) <= {0, 1}
    
    def test_process_data_inference_mode(self, sample_data):
        """Test process_data function in inference mode."""
        cat_features = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
        
        # First train the encoder and lb
        X_train, y_train, encoder, lb = process_data(
            sample_data,
            categorical_features=cat_features,
            label='salary',
            training=True
        )
        
        # Then use them for inference
        X_test, y_test, encoder_out, lb_out = process_data(
            sample_data,
            categorical_features=cat_features,
            label='salary',
            training=False,
            encoder=encoder,
            lb=lb
        )
        
        assert X_test.shape[0] == len(sample_data)
        assert y_test.shape[0] == len(sample_data)
        assert encoder_out == encoder  # Should return same encoder
        assert lb_out == lb  # Should return same lb
    
    def test_process_data_without_label(self, sample_data):
        """Test process_data function without label."""
        cat_features = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
        
        X_train, y_train, encoder, lb = process_data(
            sample_data,
            categorical_features=cat_features,
            label='salary',
            training=True
        )
        
        # Remove label column and process without label
        sample_data_no_label = sample_data.drop('salary', axis=1)
        X_test, y_test, _, _ = process_data(
            sample_data_no_label,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb
        )
        
        assert X_test.shape[0] == len(sample_data_no_label)
        assert len(y_test) == 0  # y should be empty array


class TestModelTraining:
    """Tests for model training function."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        X = np.random.rand(100, 15)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_train_model_returns_classifier(self, sample_training_data):
        """Test that train_model returns a RandomForestClassifier."""
        X, y = sample_training_data
        model = train_model(X, y)
        
        assert isinstance(model, RandomForestClassifier)
    
    def test_train_model_is_fitted(self, sample_training_data):
        """Test that the trained model is fitted."""
        X, y = sample_training_data
        model = train_model(X, y)
        
        # Check if model has been fitted
        assert hasattr(model, 'classes_')
        assert hasattr(model, 'n_features_in_')


class TestInference:
    """Tests for inference function."""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model and test data."""
        X_train = np.random.rand(100, 15)
        y_train = np.random.randint(0, 2, 100)
        model = train_model(X_train, y_train)
        
        X_test = np.random.rand(20, 15)
        
        return model, X_test
    
    def test_inference_returns_predictions(self, trained_model_and_data):
        """Test that inference returns predictions."""
        model, X_test = trained_model_and_data
        preds = inference(model, X_test)
        
        assert preds.shape[0] == X_test.shape[0]
        assert set(preds) <= {0, 1}  # Binary classification
    
    def test_inference_output_type(self, trained_model_and_data):
        """Test that inference output is a numpy array."""
        model, X_test = trained_model_and_data
        preds = inference(model, X_test)
        
        assert isinstance(preds, np.ndarray)


class TestComputeMetrics:
    """Tests for compute_model_metrics function."""
    
    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
        
        assert precision == 1.0
        assert recall == 1.0
        assert fbeta == 1.0
    
    def test_compute_metrics_all_zeros(self):
        """Test metrics when all predictions are 0."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        
        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
        
        # When all predictions are correct 0s, metrics should be 1.0
        assert precision == 1.0
        assert recall == 1.0
        assert fbeta == 1.0
    
    def test_compute_metrics_partial_accuracy(self):
        """Test metrics with partial accuracy."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 1])
        
        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
        
        # Should have some metrics calculated
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= fbeta <= 1


class TestDataSlices:
    """Tests for performance_on_categorical_slices function."""
    
    @pytest.fixture
    def sample_data_with_predictions(self):
        """Create sample data with predictions for testing slices."""
        data = pd.DataFrame({
            'category1': ['A', 'B', 'A', 'B', 'A'],
            'category2': ['X', 'X', 'Y', 'Y', 'X'],
        })
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])  # Perfect predictions
        
        return data, y_true, y_pred
    
    def test_performance_on_slices_returns_dict(self, sample_data_with_predictions):
        """Test that performance_on_categorical_slices returns a dictionary."""
        data, y_true, y_pred = sample_data_with_predictions
        
        slices = performance_on_categorical_slices(
            data, y_true, y_pred, 
            categorical_features=['category1', 'category2']
        )
        
        assert isinstance(slices, dict)
    
    def test_performance_on_slices_contains_features(self, sample_data_with_predictions):
        """Test that slices contain all features."""
        data, y_true, y_pred = sample_data_with_predictions
        
        slices = performance_on_categorical_slices(
            data, y_true, y_pred,
            categorical_features=['category1', 'category2']
        )
        
        assert 'category1' in slices
        assert 'category2' in slices
    
    def test_performance_on_slices_has_metrics(self, sample_data_with_predictions):
        """Test that slices contain required metrics."""
        data, y_true, y_pred = sample_data_with_predictions
        
        slices = performance_on_categorical_slices(
            data, y_true, y_pred,
            categorical_features=['category1']
        )
        
        for value, metrics in slices['category1'].items():
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'fbeta' in metrics
            assert 'count' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
