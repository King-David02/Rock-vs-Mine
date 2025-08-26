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
    sample_data = X[:10]
    sample_prediction = pipeline.predict(sample_data)
    signature = infer_signature(sample_data, sample_prediction)
    report_str = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report_str)
    mlflow.log_artifact("classification_report.txt")
    #mlflow.sklearn.log_model(pipeline, name="model", signature=signature)
    joblib.dump(pipeline, "pipeline.pkl")
    mlflow.log_artifact("pipeline.pkl")

with mlflow.start_run():
    mlflow.autolog()
    pipeline2 = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    prediction = pipeline2.predict(X_test)
    sample_data = X[:10]
    sample_prediction = pipeline.predict(sample_data)
    signature2 = infer_signature(sample_data, sample_prediction)
    report_str_lr = classification_report(y_test, prediction)
    with open("lr_classification_report.txt", "w") as f:
        f.write(report_str_lr)
    mlflow.log_artifact("lr_classification_report.txt")
    joblib.dump(pipeline2, "lr_model.pkl")
    mlflow.log_artifact("lr_model.pkl")
    #mlflow.sklearn.log_model(pipeline2, name="lr_model", signature=signature2)