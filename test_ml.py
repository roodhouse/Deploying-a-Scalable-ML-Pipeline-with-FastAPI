import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model


def test_inference_returns_ndarray():
    """
    Type of result: inference should return a numpy ndarray with one
    prediction per input row.
    """
    # Small fixed training set (no need for full census data)
    X_train = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]


def test_train_model_uses_random_forest():
    """
    Expected algorithm: train_model should return a fitted RandomForestClassifier.
    """
    X_train = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    # Fitted estimators expose n_features_in_
    assert model.n_features_in_ == X_train.shape[1]


def test_compute_model_metrics_known_case():
    """
    Metrics value: compute_model_metrics should return known precision,
    recall, and F1 for a fixed y / preds example.
    """

    # y:        [1, 1, 0, 0]
    # preds:    [1, 0, 0, 0]
    # TP=1, FP=0, FN=1, TN=2
    # precision=1.0, recall=0.5, f1=2/3
    
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2.0 / 3.0)
