import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    print("Loading Iris dataset...")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Feature names:
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(f"Features: {list(X.columns)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model test accuracy: {acc * 100:.2f}%")

    # Package model artifacts together to guarantee consistent feature alignment and labels
    artifacts = {
        "model": clf,
        "feature_names": list(X.columns),
        "target_names": list(target_names),
        "feature_ranges": {
            col: {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "default": float(X[col].median())
            }
            for col in X.columns
        }
    }

    # Save to disk using joblib
    joblib.dump(artifacts, "model.joblib")
    print("Artifacts saved successfully as 'model.joblib'.")

if __name__ == "__main__":
    train_and_save_model()
