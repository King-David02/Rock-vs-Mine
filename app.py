from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
with open("model/pipeline.pkl", "rb") as f:
    model = joblib.load(f)


class UserInput(BaseModel):
    Freq_1: float
    Freq_2: float
    Freq_3: float
    Freq_4: float
    Freq_5: float
    Freq_6: float
    Freq_7: float
    Freq_8: float
    Freq_9: float
    Freq_10: float
    Freq_11: float
    Freq_12: float
    Freq_13: float
    Freq_14: float
    Freq_15: float
    Freq_16: float
    Freq_17: float
    Freq_18: float
    Freq_19: float
    Freq_20: float
    Freq_21: float
    Freq_22: float
    Freq_23: float
    Freq_24: float
    Freq_25: float
    Freq_26: float
    Freq_27: float
    Freq_28: float
    Freq_29: float
    Freq_30: float
    Freq_31: float
    Freq_32: float
    Freq_33: float
    Freq_34: float
    Freq_35: float
    Freq_36: float
    Freq_37: float
    Freq_38: float
    Freq_39: float
    Freq_40: float
    Freq_41: float
    Freq_42: float
    Freq_43: float
    Freq_44: float
    Freq_45: float
    Freq_46: float
    Freq_47: float
    Freq_48: float
    Freq_49: float
    Freq_50: float
    Freq_51: float
    Freq_52: float
    Freq_53: float
    Freq_54: float
    Freq_55: float
    Freq_56: float
    Freq_57: float
    Freq_58: float
    Freq_59: float
    Freq_60: float


@app.get("/")
def welcome_page():
    return "Hi, Welcome"


@app.post("/predict")
def predict(frequencies: UserInput):
    df = pd.DataFrame([frequencies.model_dump()])
    y_pred = model.predict(df)
    return {"prediction": y_pred.tolist()}
