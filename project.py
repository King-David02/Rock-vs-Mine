import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import joblib
import dagshub
import mlflow

dagshub.init(repo_owner='King-David02', repo_name='Rock-vs-Mine', mlflow=True)

data = pd.read_csv('data/Sonar.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run():
    mlflow.autolog()
    pipeline = make_pipeline(StandardScaler(),
                            RandomForestClassifier()
                            )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report_str = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report_str)
    mlflow.log_artifact("classification_report.txt")
    mlflow.sklearn.log_model(pipeline, "model")