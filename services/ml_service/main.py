import traceback

import pandas as pd
from fastapi import FastAPI, HTTPException

from api_handler import FastAPIHandler
from pydantic import BaseModel

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / 'models' / 'model.pkl'

app = FastAPI()

handler = FastAPIHandler(model_path=MODEL_PATH)

# Pydantic модель для входных данных (признаки модели из ЛР2)
class PredictionRequest(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    high_age: int

@app.get("/")
async def read_root():
    print("### MAIN.PY LOADED ###")
    return {"Hello": "World"}


@app.post("/api/prediction/{item_id}")
async def predict(item_id: str, request: PredictionRequest):
    try:
        features_df = pd.DataFrame([{
            'age': request.age,
            'sex': request.sex,
            'cp': request.cp,
            'trestbps': request.trestbps,
            'chol': request.chol,
            'fbs': request.fbs,
            'restecg': request.restecg,
            'thalach': request.thalach,
            'exang': request.exang,
            'oldpeak': request.oldpeak,
            'slope': request.slope,
            'ca': request.ca,
            'thal': request.thal,
            'high_age': request.high_age
        }])

        # feature_names = handler.get_feature_names()
        prediction = handler.predict(features_df)


        return {
            "item_id": item_id,
            "predict": prediction
        }


    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail='Something went wrong. Check logs for more details')
