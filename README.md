FastAPI Lab 1 – Custom Iris Classifier API

This lab is based on the provided FastAPI template for exposing ML models as APIs.
I made the following custom changes and improvements to the original lab:

🔹 Modifications I Made

Custom main.py

Rewrote the API logic instead of using the base version.

Implemented an organized IrisData (request) and IrisResponse (response) using Pydantic.

Added better error handling with HTTPException.

Changed the request → response flow so prediction is clearer.

Prediction Endpoint

Modified how input features are structured before being passed to the model.

Ensured correct ordering of features: [sepal_length, sepal_width, petal_length, petal_width].

Added mapping so the model output is returned in a structured JSON.

README + Project Setup

Created a new repo with only my modified lab contents.

Cleaned .gitignore to ignore venvs, caches, and system files.

Kept iris_model.pkl so the API works without retraining.

🔹 How to Run

Install dependencies:

pip install -r requirements.txt


Train the model (optional, pre-trained model already included):

cd src
python train.py


Start the FastAPI server:

uvicorn main:app --reload


Test via Swagger UI:
👉 http://127.0.0.1:8000/docs

🔹 Example Request

POST /predict

{
  "petal_length": 2.3,
  "sepal_length": 4.0,
  "petal_width": 0.6,
  "sepal_width": 3.0
}


Response:

{
  "response": 0
}
