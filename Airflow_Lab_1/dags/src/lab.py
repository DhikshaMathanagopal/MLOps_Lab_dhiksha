import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os


CURRENT_YEAR = 2025


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features used by both training and inference."""

    engineered_df = df.copy()

    engineered_df['home_age_years'] = CURRENT_YEAR - engineered_df['year_built']
    engineered_df['per_occupant_sqft'] = engineered_df['home_size_sqft'] / (engineered_df['num_occupants'] + 0.5)

    size_denominator = np.maximum(engineered_df['home_size_sqft'] / 1000.0, 0.1)
    engineered_df['solar_coverage_ratio'] = np.nan_to_num(
        engineered_df['solar_capacity_kw'] / size_denominator,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    engineered_df['hvac_load_index'] = np.nan_to_num(
        ((engineered_df['avg_annual_temp_f'] - 62.0) * (engineered_df['home_size_sqft'] / 900.0))
        / (engineered_df['insulation_rating'] + 0.5),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    engineered_df['smart_device_intensity'] = np.nan_to_num(
        engineered_df['smart_devices_count'] / size_denominator,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    engineered_df['occupancy_pressure'] = engineered_df['num_occupants'] / (engineered_df['num_bedrooms'] + 0.5)

    return engineered_df


def load_and_validate_data():
    """
    Load the residential energy dataset, validate schema, and serialize it for downstream tasks.

    Returns:
        bytes: Serialized validated data.
    """

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))

    # Data validation
    print(f"Data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Remove rows with missing values
    df = df.dropna()

    # Validate required columns exist
    required_cols = [
        'home_size_sqft',
        'num_occupants',
        'num_bedrooms',
        'num_bathrooms',
        'year_built',
        'insulation_rating',
        'solar_capacity_kw',
        'avg_annual_temp_f',
        'smart_devices_count',
        'annual_energy_consumption_kwh',
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Validated data shape: {df.shape}")
    serialized_data = pickle.dumps(df)
    return serialized_data


def feature_engineering(data):
    """
    Perform feature engineering for the energy consumption model.

    Args:
        data (bytes): Serialized data.

    Returns:
        bytes: Serialized engineered features.
    """

    df = pickle.loads(data)
    df = add_engineered_features(df)

    feature_cols = [
        'home_size_sqft',
        'num_occupants',
        'num_bedrooms',
        'num_bathrooms',
        'insulation_rating',
        'solar_capacity_kw',
        'avg_annual_temp_f',
        'smart_devices_count',
        'home_age_years',
        'per_occupant_sqft',
        'solar_coverage_ratio',
        'hvac_load_index',
        'smart_device_intensity',
        'occupancy_pressure',
    ]

    X = df[feature_cols]
    y = df['annual_energy_consumption_kwh']

    print(f"Feature engineering complete. Features: {feature_cols}")
    print("Target variable: annual_energy_consumption_kwh")

    result = {'X': X, 'y': y, 'feature_cols': feature_cols}
    return pickle.dumps(result)


def train_model(data, filename):
    """
    Train a Random Forest regression model to predict annual energy consumption.

    Args:
        data (bytes): Serialized feature-engineered data.
        filename (str): Model filename.

    Returns:
        dict: Model metrics and scaler.
    """
    data_dict = pickle.loads(data)
    X = data_dict['X']
    y = data_dict['y']
    feature_cols = data_dict['feature_cols']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance:")
    print(f"  RMSE: {rmse:,.2f} kWh")
    print(f"  MAE: {mae:,.2f} kWh")
    print(f"  R2 Score: {r2:.4f}")
    
    # Save model and scaler
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, filename)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'feature_cols': feature_cols,
        'target': 'annual_energy_consumption_kwh'
    }
    return pickle.dumps(metrics)


def make_predictions(filename, metrics_data):
    """
    Make energy consumption predictions on the hold-out dataset using the trained model.

    Args:
        filename (str): Model filename.
        metrics_data (bytes): Serialized metrics data.

    Returns:
        dict: Prediction results.
    """
    metrics = pickle.loads(metrics_data)
    feature_cols = metrics['feature_cols']
    
    # Load model and scaler
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    scaler_path = os.path.join(os.path.dirname(__file__), "../model", "scaler.pkl")
    
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Load test data
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))

    # Feature engineering on test data
    test_df = add_engineered_features(test_df)

    # Select features
    X_test = test_df[feature_cols]
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['predicted_energy_kwh'] = predictions

    print(f"Made predictions on {len(test_df)} test samples")
    print(
        "Predicted energy range: "
        f"{predictions.min():,.2f} kWh - {predictions.max():,.2f} kWh"
    )
    print(f"Average predicted energy: {predictions.mean():,.2f} kWh")

    # Save predictions
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    results_path = os.path.join(output_dir, "energy_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

    return pickle.dumps({
        'predictions': predictions.tolist(),
        'avg_prediction_kwh': float(predictions.mean())
    })