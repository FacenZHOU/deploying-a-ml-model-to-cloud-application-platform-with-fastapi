"""
this modiule contains unit tests for the model rf in model.py

Author: Facen
Date: 2023-04-05
"""
from pathlib import Path
import logging
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from model import train_model, compute_model_metrics


def test_model_type():
    """
    Test the type of the model.
    """
    X_train1 = np.random.rand(100, 10)
    y_train1 = np.random.randint(0, 2, 100)
    model1 = train_model(X_train1, y_train1)
    assert type(model1).__name__ == 'RandomForestClassifier'


def test_model_shape():
    """
    Test the shape of the model.
    """
    X_train2 = np.random.rand(100, 10)
    y_train2 = np.random.randint(0, 2, 100)
    model2 = train_model(X_train2, y_train2)
    assert model2.shape == (100, 10)


def test_output_type():
    """
    Test the type of the output.
    """
    y1 = np.random.randint(0, 2, 10)
    preds1 = np.random.randint(0, 2, 10)
    precision1, recall1, fbeta1 = compute_model_metrics(y1, preds1)
    assert isinstance(precision1, float)
    assert isinstance(recall1, float)
    assert isinstance(fbeta1, float)       