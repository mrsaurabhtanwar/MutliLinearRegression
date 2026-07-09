import joblib
import numpy as np
import time
import logging


from sqlalchemy import create_engine, Column, Integer, DateTime, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title='House price prediction')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


engine = create_engine("sqlite:///house-price-prediction.db", connect_args={"check_same_thread":False})
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PricePre(Base):
    __tablename__ = "House-Price-Prediction"
    id = Column(Integer, primary_key=True)
    income = Column(Float, nullable=False)
    house_age = Column(Integer, nullable=False)
    rooms = Column(Integer, nullable=False)
    area_population = Column(Integer, nullable=False)
    members = Column(Integer, nullable=False)
    longitude = Column(Float, nullable=False)
    datetime = Column(DateTime, nullable=False)
    latency = Column(Float, nullable=False)
    error = Column(String)
    
Base.metadata.create_all(engine)

db = sessionLocal()

def get_db():
    try:
        yield db 
    finally:
        db.close()


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
def predication(data: HousingInput, db: Session = Depends(get_db)):
    current_dt = datetime.now()
    start_time = time.time()
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

        latency = time.time() - start_time
        db.add(
             PricePre(
                income=data.income,
                house_age=data.house_age,
                rooms=data.rooms,
                area_population=data.area_population,
                members=data.members,
                longitude=data.longitude,
                datetime=current_dt,
                latency=latency,
                error=None
            )
        )
        db.commit()


        return {
            "predicted_price": round(predicted_price, 2) 
        }
    except Exception as e:
        db.rollback()
        db.add(PricePre(
                 income=data.income,
                 house_age=data.house_age,
                 rooms=data.rooms,
                 area_population=data.area_population,
                 members=data.members,
                 longitude=data.longitude,
                 datetime=current_dt,
                 latency=0,
                 error=str(e)))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")