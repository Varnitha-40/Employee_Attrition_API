Employee Attrition Prediction System

Project Overview:
This project predicts whether an employee is likely to remain active in the organization or leave the company. The goal was not only to build a machine learning model but also to deploy it as a complete application with an API, monitoring, logging, and a user-friendly dashboard.

What I Did:
Cleaned and prepared the employee attrition dataset
Encoded categorical features using Label Encoding
Trained and evaluated multiple machine learning models
Selected a Decision Tree Classifier for prediction
Saved the trained model using Pickle
Built a FastAPI backend to expose the model as an API
Created a Gradio dashboard for user interaction
Added audit logging to store prediction history
Added monitoring to track API usage
Implemented input validation (guardrails) to handle invalid data
Displayed prediction probabilities and feature importance
Technologies Used
Python
Pandas
Scikit-learn
FastAPI
Gradio
Pickle
API Endpoints
GET /

Returns a simple message indicating that the API is running.

POST /predict

Accepts employee details and returns:

Prediction (ACTIVE / TERMINATED)
Prediction probability
GET /metrics

Returns the total number of predictions made through the API.

Dashboard Features

The Gradio dashboard allows users to:

Enter employee information
Submit prediction requests
View prediction results
View prediction confidence
Visualize feature importance
Project Architecture

Gradio Dashboard → FastAPI API → Decision Tree Model → Prediction

Additional components:

Audit Logging (CSV)
Monitoring Endpoint
Input Validation
Learning Outcomes

Through this project, I learned:

Machine Learning model deployment
API development with FastAPI
Building interactive dashboards with Gradio
Monitoring and logging in ML applications
End-to-end ML workflow from training to deployment
Author

Sreevarnitha
