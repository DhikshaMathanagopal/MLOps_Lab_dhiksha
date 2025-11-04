# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_and_validate_data,
    feature_engineering,
    train_model,
    make_predictions
)

from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'ml_engineer',
    'start_date': datetime(2025, 1, 15),
    'retries': 1,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance for the home energy consumption pipeline with the defined default arguments
dag = DAG(
    'Home_Energy_Consumption',
    default_args=default_args,
    description='ML pipeline for residential energy consumption forecasting with validation, feature engineering, and evaluation',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
    tags=['ml', 'regression', 'energy'],
)

# Task 1: Load and validate data
load_validate_task = PythonOperator(
    task_id='load_and_validate_data',
    python_callable=load_and_validate_data,
    dag=dag,
)

# Task 2: Feature engineering - create new features from raw data
feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    op_args=[load_validate_task.output],
    dag=dag,
)

# Task 3: Train model - train Random Forest regressor and evaluate
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_args=[feature_engineering_task.output, "home_energy_model.pkl"],
    dag=dag,
)

# Task 4: Make predictions on test data
predict_task = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    op_args=["home_energy_model.pkl", train_model_task.output],
    dag=dag,
)

# Set task dependencies - linear pipeline
load_validate_task >> feature_engineering_task >> train_model_task >> predict_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()