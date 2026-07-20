import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import inference, train_model


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


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
