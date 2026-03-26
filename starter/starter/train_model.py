"""
Model Training Script

This module trains a Random Forest classifier on census income data.
It performs the following steps:
1. Load and split the cleaned census data
2. Process categorical features using one-hot encoding
3. Train a Random Forest model
4. Generate predictions and compute performance metrics
5. Analyze model performance on categorical data slices
6. Save the trained model, encoder, and label binarizer for inference.

The script evaluates performance using precision, recall, and F-beta score.
It also generates a detailed report of model performance on categorical
feature slices to assess fairness and identify potential biases.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    performance_on_categorical_slices,
)

# Load the cleaned data
data = pd.read_csv("../data/census_cleaned.csv")

# Optional: use K-fold cross validation instead of train-test split.
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
train_precision, train_recall, train_fbeta = compute_model_metrics(
    y_train, y_train_preds
)
test_precision, test_recall, test_fbeta = compute_model_metrics(
    y_test, y_test_preds
)

print("Model Training Complete!")
print("\nTraining Metrics:")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall: {train_recall:.4f}")
print(f"  F-Beta: {train_fbeta:.4f}")

print("\nTest Metrics:")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F-Beta: {test_fbeta:.4f}")

# Compute and save slice metrics
print("\nGenerating model performance on categorical slices...")
slice_metrics = performance_on_categorical_slices(
    test, y_test, y_test_preds, cat_features
)

# Write slice metrics to file
with open("slice_output.txt", "w") as f:
    # Header
    f.write("=" * 100 + "\n")
    f.write("MODEL PERFORMANCE ON CATEGORICAL DATA SLICES\n")
    f.write("=" * 100 + "\n\n")

    # Overall metrics
    f.write("OVERALL TEST SET METRICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall:    {test_recall:.4f}\n")
    f.write(f"F-Beta:    {test_fbeta:.4f}\n")
    f.write(f"Total Samples: {len(y_test)}\n\n")

    # Slice metrics
    f.write("PERFORMANCE BY CATEGORICAL FEATURE\n")
    f.write("=" * 100 + "\n\n")

    for feature in sorted(slice_metrics.keys()):
        f.write(f"{feature.upper()}\n")
        f.write("-" * 100 + "\n")
        f.write(
            f"{'Feature Value':<30} {'Precision':<15} {'Recall':<15} "
            f"{'F-Beta':<15} {'Count':<10}\n"
        )
        f.write("-" * 100 + "\n")

        # Sort values by count (descending) for readability
        values = sorted(
            slice_metrics[feature].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )

        for value, metrics in values:
            f.write(
                f"{str(value):<30} {metrics['precision']:<15.4f} "
                f"{metrics['recall']:<15.4f} {metrics['fbeta']:<15.4f} "
                f"{metrics['count']:<10}\n"
            )

        f.write("\n")

    # Summary statistics
    f.write("=" * 100 + "\n")
    f.write("ANALYSIS SUMMARY\n")
    f.write("-" * 100 + "\n")

    total_slices = sum(len(values) for values in slice_metrics.values())
    f.write(f"Total categorical features analyzed: {len(slice_metrics)}\n")
    f.write(f"Total feature slices: {total_slices}\n")

    # Find performance extremes
    all_precisions = []
    all_recalls = []
    all_fbetas = []

    for feature_values in slice_metrics.values():
        for metrics in feature_values.values():
            all_precisions.append(metrics["precision"])
            all_recalls.append(metrics["recall"])
            all_fbetas.append(metrics["fbeta"])

    if all_precisions:
        min_prec = min(all_precisions)
        max_prec = max(all_precisions)
        f.write(f"\nPrecision Range: {min_prec:.4f} - {max_prec:.4f}\n")
        min_rec = min(all_recalls)
        max_rec = max(all_recalls)
        f.write(f"Recall Range: {min_rec:.4f} - {max_rec:.4f}\n")
        min_fb = min(all_fbetas)
        max_fb = max(all_fbetas)
        f.write(f"F-Beta Range: {min_fb:.4f} - {max_fb:.4f}\n")

    msg = "\nNote: Performance variations across slices indicate potential "
    msg += "fairness considerations.\n"
    f.write(msg)
    f.write("=" * 100 + "\n")

print("✓ Slice metrics saved to slice_output.txt")

# Save the model, encoder, and label binarizer
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("../model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

print("\nModel, encoder, and label binarizer saved to model/ directory!")
