"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

import os
import joblib
import pandas as pd

# TODO 1: Load your serialized churn model (and preprocessor if any) from data/
MODEL_PATH = os.path.join("data", "model.pkl")
PREPROCESSOR_PATH = os.path.join("data", "column_transformer.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def preprocess(features: list[float]) -> list[float]:
    """
    Takes raw features and applies necessary preprocessing (e.g. scaling).
    """
    # TODO 2: Apply any preprocessing steps here (if applicable)
    # features_2d = np.array(features).reshape(1, -1)
    column_names = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    # Convert the 1D list into a 2D Pandas DataFrame row with names
    features_2d = pd.DataFrame([features], columns=column_names)

    if preprocessor is not None:
        return preprocessor.transform(features_2d)
    return features_2d


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of raw feature values and returns a churn prediction (0 or 1).
    """
    # TODO 3: Preprocess the features
    processed_features = preprocess(features)

    # TODO 4: Use model.predict() on processed_features to get a prediction and return it as an int
    #         Hint: model.predict() expects a 2D array
    prediction = model.predict(processed_features)
    # Extract the scalar item from the prediction array and convert to a native int
    return int(prediction[0])


if __name__ == "__main__":
    # TODO 5: Replace with sample features that match your model
    sample = sample = [
        608.0,
        "Spain",
        "Female",
        41.0,
        1.0,
        83807.86,
        1.0,
        0.0,
        1.0,
        112542.58,
    ]
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
