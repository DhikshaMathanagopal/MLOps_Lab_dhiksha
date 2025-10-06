from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Docker Lab 1 - Dhiksha Version")

# Load trained model
model_path = "src/iris_model.pkl"
model = joblib.load(model_path)

# Define request body for prediction
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Welcome to Dhiksha's Docker Lab 1!"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "owner": "Dhiksha Mathanagopal",
        "version": "1.1-dhiksha",
        "app_mode": os.getenv("APP_MODE", "dev")
    }

@app.post("/predict")
def predict(data: IrisFeatures):
    features = [[
        data.sepal_length, data.sepal_width,
        data.petal_length, data.petal_width
    ]]
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
