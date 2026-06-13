from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import csv
from datetime import datetime

app = FastAPI()

prediction_count = 0

# Load model
with open("attrition_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input schema
class EmployeeData(BaseModel):
    age: int
    length_of_service: int
    city_name: int
    department_name: int
    job_title: int
    store_name: int
    gender_short: int
    gender_full: int
    STATUS_YEAR: int
    BUSINESS_UNIT: int


@app.get("/")
def home():
    return {"message": "Employee Attrition Prediction API is running"}


@app.post("/predict")
def predict(data: EmployeeData):

    # guardrails
    if data.age < 18 or data.age > 65:
        return {"error": "Age must be between 18 and 65"}

    if data.length_of_service < 0 or data.length_of_service > 40:
        return {"error": "Length of service must be between 0 and 40"}

    global prediction_count
    prediction_count += 1

    input_data = pd.DataFrame([[

        data.age,
        data.length_of_service,
        data.city_name,
        data.department_name,
        data.job_title,
        data.store_name,
        data.gender_short,
        data.gender_full,
        data.STATUS_YEAR,
        data.BUSINESS_UNIT

    ]], columns=[
        'age',
        'length_of_service',
        'city_name',
        'department_name',
        'job_title',
        'store_name',
        'gender_short',
        'gender_full',
        'STATUS_YEAR',
        'BUSINESS_UNIT'
    ])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    result = "ACTIVE" if prediction == 0 else "TERMINATED"

    # audit log
    with open("audit_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(),
            data.age,
            data.length_of_service,
            data.city_name,
            data.department_name,
            data.job_title,
            data.store_name,
            data.gender_short,
            data.gender_full,
            data.STATUS_YEAR,
            data.BUSINESS_UNIT,
            result
        ])

    return {
        "prediction": result,
        "probability": {
            "ACTIVE": float(proba[0]),
            "TERMINATED": float(proba[1])
        }
    }


@app.get("/metrics")
def metrics():
    return {
        "total_predictions": prediction_count
    }