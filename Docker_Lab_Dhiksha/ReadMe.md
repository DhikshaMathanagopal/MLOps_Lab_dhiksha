Docker Lab 1 – Personalized Version by Dhiksha Mathanagopal
Overview

This project demonstrates how to containerize a Machine Learning model using Docker and serve it via a FastAPI REST interface.
The app trains a Logistic Regression model on the Iris dataset, exposes endpoints for health checks and predictions, and runs fully inside a Docker container.

Key Personal Modifications by Dhiksha

Replaced the default RandomForestClassifier with a Logistic Regression model (new model logic and training workflow).
Integrated FastAPI instead of Flask — with /, /health, and /predict endpoints.
Added a structured Pydantic model (IrisFeatures) for type-safe input validation.
Personalized endpoint metadata and responses (owner, version, model_type, student_version).
Built and exposed the application through Uvicorn ASGI server inside Docker.
Verified model inference and container behavior using Swagger UI and cURL tests.
Enhanced readability, naming, and structure of code + Dockerfile for modular maintainability.

Build & Run Instructions
Build and run the Docker container:
docker build -t docker_lab_dhiksha .
docker run -p 8000:8000 docker_lab_dhiksha

Once started, open:
Home: http://localhost:8000
Health Check: http://localhost:8000/health
Swagger UI: http://localhost:8000/docs