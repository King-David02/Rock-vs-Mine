import os
import dagshub
import joblib
import mlflow
import pandas as pd
import psycopg
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from evidently.sdk.panels import PanelMetric, bar_plot_panel, text_panel
from evidently.ui.workspace import Workspace
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

create_table = """
drop table if exists metrics;
create table metrics(
drifted_column_count FLOAT,
Accuracy FLOAT,
F1Score FLOAT);
"""

dagshub.init(repo_owner="King-David02", repo_name="Rock-vs-Mine", mlflow=True)


@task
def load_data(file_path: str):
    data = pd.read_csv(file_path)
    X = data.drop(["Target"], axis=1)
    y = data["Target"]
    return X, y, data


@task
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@task
def train_random_forest(X_train, X_test, y_train, y_test, data, output_path):
    with mlflow.start_run(run_name="Random Forest"):
        mlflow.autolog()
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
        pipeline = pipeline.fit(X_train, y_train)
        with open("model/pipeline.pkl", "wb") as f:
            joblib.dump(pipeline, f)
        y_pred = pipeline.predict(X_test)
        data["prediction"] = pipeline.predict(data.drop(["Target"], axis=1))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
            mlflow.log_artifact("classification_report.txt")
        return pipeline, data


@task
def other_data_prediction(data_path, pipeline, output_path):
    other_data = pd.read_csv(data_path)
    other_data["prediction"] = pipeline.predict(other_data.drop(["Target"], axis=1))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    other_data.to_csv(output_path, index=False)
    return other_data


@task
def evidently_report(data, other_data):
    definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="Target", prediction_labels="prediction", pos_label="M"
            )
        ]
    )

    reff_data = Dataset.from_pandas(data=data, data_definition=definition)
    curr_data = Dataset.from_pandas(data=other_data, data_definition=definition)

    report = Report(
        metrics=[DataDriftPreset(), ClassificationPreset()], include_tests=True
    )

    report = report.run(current_data=curr_data, reference_data=reff_data)
    result = report.dict()
    return report, result


@task
def evidently_dashboard(report):
    ws = Workspace.create("Workspace")
    project = ws.create_project("Rock vs Mine Dashboard")
    ws.add_run(project.id, report)

    project.dashboard.add_panel(text_panel(title="Rock vs Mine Dashboard"))

    project.dashboard.add_panel(
        bar_plot_panel(
            title="Accuracy Visualization",
            values=[
                PanelMetric(
                    legend="ANI",
                    metric="Accuracy",
                )
            ],
            size="half",
        )
    )


@task
def metrics_to_postgres(result):
    conn = psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    )

    res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    if len(res.fetchall()) == 0:
        conn.execute("CREATE database test;")
        conn = psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example",
            autocommit=True,
        )

    else:
        print("database exists already")
        conn = psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example",
            autocommit=True,
        )

    conn.execute(create_table)

    drifted_column_count = result["metrics"][0]["value"]["count"]
    Accuracy = result["metrics"][64]["value"]
    F1Score = result["metrics"][67]["value"]
    with conn.cursor() as cur:
        cur.execute(
            "insert into metrics(drifted_column_count, Accuracy, F1Score) values (%s, %s, %s)",
            (drifted_column_count, Accuracy, F1Score),
        )


@flow(name="Rock vs Mne Flow")
def rock_mine_pipeline(file_path="data/Sonar.csv"):
    other_data_path = "data/rock_vs_mine_dataset.csv"
    data_output = "data/predicted_data/data.csv"
    other_data_output = "data/predicted_data/otherdata.csv"
    X, y, data = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline, data = train_random_forest(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        data=data,
        output_path=data_output,
    )
    other_data = other_data_prediction(
        data_path=other_data_path, pipeline=pipeline, output_path=other_data_output
    )
    report, result = evidently_report(data=data, other_data=other_data)
    evidently_dashboard(report)
    metrics_to_postgres(result)


if __name__ == "__main__":
    rock_mine_pipeline()
