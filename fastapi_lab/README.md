# FastAPI Lab 1 – Custom Iris Classifier API

This lab is based on the provided FastAPI template for exposing ML models as APIs.  
I made the following **custom modifications** to ensure the work is unique and goes beyond the starter code.

---

## Modifications I Made

### Model Changes
- Replaced **DecisionTreeClassifier** with **RandomForestClassifier** for training (`train.py`).
- Added **species mapping** so predictions return `"setosa"`, `"versicolor"`, `"virginica"` instead of raw integers.
- Extended `/predict` response with **probabilities** (confidence scores).
- Kept `iris_model.pkl` so the API loads the trained model without retraining each time.

### API Improvements
- Rewrote the `main.py` logic to be structured and clear.
- Implemented `IrisData` (request schema) and `IrisResponse` (response schema) using **Pydantic**.
- Added better **error handling** with `HTTPException`.
- Changed the request → response flow so predictions are clearer and structured JSON is returned.
- Ensured correct **ordering of features**: `[sepal_length, sepal_width, petal_length, petal_width]`.
- Added a new endpoint `/species_map` to expose the mapping of class → species.

### Project Setup
- Created a new repo with **only my modified lab contents**.
- Cleaned `.gitignore` to ignore venv, caches, and system files.
- Added a focused **README.md** documenting all changes.

---

## Project Structure
mlops_labs/
└── fastapi_lab1
├── assets/
├── fastapi_lab1_env/ <- virtual environment (ignored in git)
├── model/
│ └── iris_model.pkl
├── src/
│ ├── init.py
│ ├── data.py
│ ├── main.py
│ ├── predict.py
│ └── train.py
├── README.md
└── requirements.txt

yaml
Copy code

---

## How to Run

1. Create and activate the virtual environment:
   ```bash
   python3 -m venv fastapi_lab1_env
   source fastapi_lab1_env/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the API:

bash
Copy code
cd src
uvicorn main:app --reload
Open the API docs in browser:

arduino
Copy code
http://127.0.0.1:8000/docs
✅ Example Request
Input
json
Copy code
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Output
json
Copy code
{
  "response": "setosa",
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.00
  }
}
