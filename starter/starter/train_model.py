"""
Script to train machine learning model.

Authoir: Facen
Date: 2023-04-05
"""

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import logging
from ml.model import train_model, compute_model_metrics, inference
import joblib
import pandas as pd
from ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

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

# Add code to load in the data.
logging.info("Importing data")
data = pd.read_csv("../data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Proces the test data with the process_data function.
logging.info("Preprocessing data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
     training=False, encoder=encoder, lb=lb)

# Train and save a model.
logging.info("Training model")
model = train_model(X_train, y_train)

# Compute the model's accuracy.
logging.info("Computing model accuracy")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save the models.
logging.info("Saving models")
joblib.dump(model, "../model/model.pkl")
joblib.dump(encoder, "../model/encoder.pkl")
joblib.dump(lb, "../model/lb.pkl")

