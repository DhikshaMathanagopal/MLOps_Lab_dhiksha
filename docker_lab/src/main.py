from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = FastAPI(title="Docker Lab 1 - Dhiksha Version")

MODEL_PATH = "iris_model.pkl"

# Train and save model if not exists
if not os.path.exists(MODEL_PATH):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ✅ Changed model from RandomForest to Logistic Regression
    model = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto")
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "Welcome to Dhiksha's Docker Lab 1 - Logistic Regression Edition!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "owner": "Dhiksha Mathanagopal",
        "model_type": "Logistic Regression",
        "version": "2.0-dhiksha",
        "app_mode": os.getenv("APP_MODE", "student_version")
    }


@app.post("/predict")
def predict(data: IrisFeatures):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
