from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_and_save_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train RandomForest instead of DecisionTree
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X, y)

    # Save model to ../model/
    joblib.dump(model, "../model/iris_model.pkl")
    print("✅ RandomForest model trained and saved as iris_model.pkl")


if __name__ == "__main__":
    train_and_save_model()
