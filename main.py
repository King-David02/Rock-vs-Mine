import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import joblib
from prefect import task, flow
from typing import Tuple, Any
import os


dagshub.init(repo_owner='King-David02', repo_name='Rock-vs-Mine', mlflow=True)

@task
def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare the dataset"""
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

@task
def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets"""
    print(f"Splitting data with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

@task
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, X: np.ndarray) -> dict:
    """Train Random Forest model with MLflow tracking"""
    print("Training Random Forest model...")
    
    with mlflow.start_run(run_name='Random Forest'):
        mlflow.autolog()
        
        pipeline = make_pipeline(StandardScaler(),
                                  RandomForestClassifier())
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        sample_data = X[:10]
        sample_prediction = pipeline.predict(sample_data)
        signature = infer_signature(sample_data, sample_prediction)
        
        report_str = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report_str)
        mlflow.log_artifact("classification_report.txt")
        
        print("Random Forest training completed!")
        return {
            'model': pipeline,
            'predictions': y_pred,
            'signature': signature,
            'report': report_str
        }

@task
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, X: np.ndarray) -> dict:
    """Train Logistic Regression model with MLflow tracking"""
    print("Training Logistic Regression model...")
    
    with mlflow.start_run(run_name='Logistic Regression'):
        mlflow.autolog()
        

        pipeline = make_pipeline(StandardScaler(),
                                  LogisticRegression())
        pipeline.fit(X_train, y_train)
        
        prediction = pipeline.predict(X_test)
        
        sample_data = X[:10]
        sample_prediction = pipeline.predict(sample_data)
        signature = infer_signature(sample_data, sample_prediction)
        
        report_str = classification_report(y_test, prediction)
        with open("lr_classification_report.txt", "w") as f:
            f.write(report_str)
        mlflow.log_artifact("lr_classification_report.txt")
        
        print("Logistic Regression training completed!")
        return {
            'model': pipeline,
            'predictions': prediction,
            'signature': signature,
            'report': report_str
        }

@task
def compare_models(rf_results: dict, lr_results: dict) -> dict:
    """Compare the performance of both models"""
    print("\n=== Model Comparison ===")
    print("Random Forest Report:")
    print(rf_results['report'])
    print("\nLogistic Regression Report:")
    print(lr_results['report'])
    
    return {
        'random_forest': rf_results,
        'logistic_regression': lr_results,
        'comparison_complete': True
    }

@flow(name="Rock vs Mine Classification Pipeline")
def ml_pipeline(data_path: str = 'data/Sonar.csv'):
    """
    Main ML pipeline flow for Rock vs Mine classification
    
    This flow orchestrates the entire ML process:
    1. Load data
    2. Split into train/test
    3. Train Random Forest model
    4. Train Logistic Regression model  
    5. Compare results
    """
    print("🚀 Starting Rock vs Mine ML Pipeline...")
    
    X, y = load_data(data_path)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    rf_results = train_random_forest(X_train, y_train, X_test, y_test, X)
    lr_results = train_logistic_regression(X_train, y_train, X_test, y_test, X)
    
    final_results = compare_models(rf_results, lr_results)
    
    print("✅ Pipeline completed successfully!")
    return final_results

if __name__ == "__main__":
    result = ml_pipeline()