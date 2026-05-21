"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from litestar import Litestar, get, post
from pydantic import BaseModel

from app.model_utils import predict_churn

from app.logger_setup import setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    # TODO 1: Add one field (type float) per feature your model expects
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# TODO 2: Create a GET endpoint at "/" that returns a welcome message
#         Log that the home endpoint was accessed
@get("/")
async def home() -> dict:
    logger.info("Home endpoint was accessed.")
    return {"message": "Welcome to the Bank Customer Churn Prediction API"}


# TODO 3: Create a GET endpoint at "/health" that returns {"status": "healthy"}
@get("/health")
async def health() -> dict:
    logger.info("Health check endpoint was accessed.")
    return {"status": "healthy"}


# TODO 4: Create a POST endpoint at "/predict" that:
#         - Accepts a ChurnRequest as the data parameter
#         - Extracts features into a list
#         - Calls predict_churn(features)
#         - Returns the prediction
#         - Logs the input features and the prediction result
@post("/predict")
async def predict(data: ChurnRequest) -> dict:
    # Extract features into a dictionary
    features_dict = data.model_dump()

    # Extract features into a list in the precise order your model expects them
    features_list = [
        features_dict["CreditScore"],
        features_dict["Geography"],
        features_dict["Gender"],
        features_dict["Age"],
        features_dict["Tenure"],
        features_dict["Balance"],
        features_dict["NumOfProducts"],
        features_dict["HasCrCard"],
        features_dict["IsActiveMember"],
        features_dict["EstimatedSalary"],
    ]

    logger.info(f"Prediction input features: {features_list}")

    # Run prediction logic from model_utils
    prediction = predict_churn(features_list)

    logger.info(f"Prediction result: {prediction}")

    return {"prediction": prediction}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
# TODO 5: Register your endpoint functions in the list below
app = Litestar(
    route_handlers=[home, health, predict],
)
