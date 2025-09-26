from main import split_data, load_data, train_random_forest, other_data_prediction
import numpy as np
import pandas as pd
import pytest
import tempfile

X = np.random.rand(100, 60)
y = np.array(['R', 'M'] * 50)

data = {f'feature{i+1}': np.random.rand(100) for i in range(60)}
data['Target'] = ['R'] * (100 // 2) + ['M'] * (100 // 2)
test_data = pd.DataFrame(data)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
data = test_data.to_csv(index=False)

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    test_data.to_csv(tmp.name, index=False)
    file_path = tmp.name

def test_load_data():
    X, y, data = load_data(file_path)
    assert X.shape == (100, 60)
    assert y.shape == (100,)
    assert data.shape == (100, 61)
    assert set(y).issubset({'R', 'M'})
    

def test_train_test_split():
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    #assert X_train[1] == X_test[1]
    assert len(X_train) + len(X_test) == 100
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20
    return X_train, X_test, y_train, y_test

def test_train_random_forest():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline,_ = train_random_forest(X_train, X_test, y_train, y_test, test_data, tmp_path)
    y_pred = pipeline.predict(X_test)
    assert len(y_pred) == len(X_test)

def train_other_data_prediction():
    pass