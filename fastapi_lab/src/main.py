from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import List
from predict import predict_data

app = FastAPI(title="Iris Classifier API", version="1.1.0")

# Map prediction classes to species names
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}


# Input model
class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float


# Response model (now includes probabilities)
class IrisResponse(BaseModel):
    class_id: int
    species: str
    probabilities: List[float]


# Health check endpoint
@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        # Arrange features in correct order for model
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]]

        # Predict class id
        prediction = predict_data(features)
        class_id = int(prediction[0])

        # Import model from predict.py so we can access probabilities
        from predict import model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0].tolist()
        else:
            probs = []

        return IrisResponse(
            class_id=class_id,
            species=species_map[class_id],
            probabilities=probs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint to expose species mapping
@app.get("/species_map", status_code=status.HTTP_200_OK)
async def get_species_map():
    """
    Returns the mapping between class_id and species name.
    """
    return species_map
