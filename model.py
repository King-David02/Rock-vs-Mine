import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import joblib
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from evidently import Dataset, DataDefinition
from evidently import Report
from evidently import BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.sdk.panels import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import psycopg

print("STARTING")
create_table = """
drop table if exists metrics;
create table metrics(
drifted_column_count float,
Accuracy float,
F1Score float
);
"""
print("CONNECTING TO POSTGRES SERVER")
conn = psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True)
print("CONNECTED TO SERVER")

res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
if len(res.fetchall()) == 0:
    conn.execute('CREATE database test;')
    print("DATABASE CREATED")
    conn = psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True)


else:
    conn = psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True)
conn.execute(create_table)



#dagshub.init(repo_owner='King-David02', repo_name='Rock-vs-Mine', mlflow=True)
reference_data = pd.read_csv('data/predicted_data/reference_data.csv')
current_data = pd.read_csv('data/predicted_data/current_data.csv')
print('DATA READ SUCCESSFUL')





definition = DataDefinition(classification=[
    BinaryClassification(
        target='target_encoded',
        prediction_labels='prediction',
        pos_label=0
    )
])

reff_data = Dataset.from_pandas(data=reference_data, data_definition=definition)
curr_data = Dataset.from_pandas(data=current_data, data_definition=definition)


report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])
report = report.run(current_data=reff_data, reference_data=curr_data)
print("METRICS REPORT DONE")

result = report.dict()

drifted_column_count = result['metrics'][0]['value']['count']
Accuracy = result['metrics'][64]['value']
F1Score = result['metrics'][67]['value']
curr = conn.cursor()
curr.execute(
    "insert into metrics(drifted_column_count, Accuracy, F1Score) values (%s, %s, %s)",
    (drifted_column_count, Accuracy, F1Score)
)
print('DATA SENT SUCCESSFULLY')
