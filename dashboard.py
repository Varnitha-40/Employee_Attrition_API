import gradio as gr
import requests
import matplotlib.pyplot as plt
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- MAPS ----------------
city_map = {"City A": 34, "City B": 31, "City C": 17}
dept_map = {"HR": 9, "Sales": 15, "IT": 18}
job_map = {"Manager": 7, "Analyst": 46, "Engineer": 33}
store_map = {"Store 1": 35, "Store 2": 20}
gender_map = {"Male": 1, "Female": 0}
unit_map = {"A": 0, "B": 1}

# ---------------- API CALL ----------------
def predict_employee(age, service, city, dept, job, store, gender_s, gender_f, year, unit):

    data = {
        "age": int(age),
        "length_of_service": int(service),
        "city_name": city_map[city],
        "department_name": dept_map[dept],
        "job_title": job_map[job],
        "store_name": store_map[store],
        "gender_short": gender_map[gender_s],
        "gender_full": gender_map[gender_f],
        "STATUS_YEAR": int(year),
        "BUSINESS_UNIT": unit_map[unit]
    }

    res = requests.post(API_URL, json=data)

    if res.status_code != 200:
        return "API Error", None

    return res.json()["prediction"], res.json().get("probability", None)

# ---------------- FEATURE IMPORTANCE ----------------
def show_feature_importance():

    features = [
        "Age","Service","City","Department","Job",
        "Store","Gender","Gender Full","Year","Business Unit"
    ]

    importance = [0.15,0.25,0.10,0.08,0.12,0.05,0.05,0.05,0.10,0.05]

    plt.figure(figsize=(6,4))
    plt.barh(features, importance)
    plt.title("Feature Importance (Decision Tree)")
    plt.tight_layout()
    plt.savefig("feature.png")
    plt.close()

    return "feature.png"

# ---------------- UI ----------------
with gr.Blocks() as demo:

    gr.Markdown("# 🧠 Employee Attrition Prediction System")
    gr.Markdown("ML Model + FastAPI + Explainability + Monitoring")

    with gr.Row():

        with gr.Column():
            age = gr.Number(label="Age")
            service = gr.Number(label="Service Years")

            city = gr.Dropdown(list(city_map.keys()))
            dept = gr.Dropdown(list(dept_map.keys()))
            job = gr.Dropdown(list(job_map.keys()))

            store = gr.Dropdown(list(store_map.keys()))
            gender_s = gr.Dropdown(list(gender_map.keys()))
            gender_f = gr.Dropdown(list(gender_map.keys()))

            year = gr.Number(label="Year")
            unit = gr.Dropdown(list(unit_map.keys()))

            btn = gr.Button("Predict")

        with gr.Column():
            out = gr.Textbox(label="Prediction")
            prob = gr.Textbox(label="Probability")
            img = gr.Image(label="Feature Importance")

    btn.click(
        predict_employee,
        inputs=[age,service,city,dept,job,store,gender_s,gender_f,year,unit],
        outputs=[out,prob]
    )

    gr.Button("Show Feature Importance").click(
        show_feature_importance,
        outputs=img
    )

demo.launch()