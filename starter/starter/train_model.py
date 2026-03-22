# Script to train machine learning model.

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Load the cleaned data
data = pd.read_csv("../data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Generate predictions
y_train_preds = inference(model, X_train)
y_test_preds = inference(model, X_test)

# Compute metrics
train_precision, train_recall, train_fbeta = compute_model_metrics(y_train, y_train_preds)
test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_test_preds)

print("Model Training Complete!")
print(f"\nTraining Metrics:")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall: {train_recall:.4f}")
print(f"  F-Beta: {train_fbeta:.4f}")

print(f"\nTest Metrics:")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F-Beta: {test_fbeta:.4f}")

# Save the model, encoder, and label binarizer
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("../model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

print("\nModel, encoder, and label binarizer saved to model/ directory!")
