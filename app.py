import joblib
import numpy as np
import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title='House price prediction')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    model = joblib.load('mutli_regressor_model.joblib')
    scaler = joblib.load('Standard_scaler_multi.joblib')
except Exception as e:
    raise RuntimeError(f"Failed to load the model artificates : {e}")

class HousingInput(BaseModel):
    income : float
    house_age : float
    rooms : float
    area_population : float
    members : float
    longitude : float


@app.post("/predict-house-price")
def predication(data: HousingInput):
    try:
        scaled_input = scaler.transform([[data.income, 
                            data.house_age, 
                            data.rooms,
                            data.area_population,
                            data.members,
                            data.longitude
                            ]])
        
        predication = model.predict(scaled_input)
        predicted_price = predication[0]

        return {
            "predicted_price": round(predicted_price, 2) 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")