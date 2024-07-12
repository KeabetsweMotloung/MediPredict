import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__),'..', 'Model', 'Medical_model_gradientR.joblib')

medical_model = joblib.load(model_path)


@app.route('/')
def LandingPage():
    return "Welcome to the MediPredict"


@app.route('/predict',methods=['POST'])
def predict():
   # Get user input from POST request
    input_data = request.get_json()

    # Extract features from user input
    age = input_data['age']
    sex = 1 if input_data['sex'].lower() == 'female' else 0
    bmi = input_data['bmi']
    children = input_data['children']
    smoker = 1 if input_data['smoker'].lower() == 'yes' else 0
    region = input_data['region']

    client_data = np.array([[age, sex, bmi, children, smoker, region]])

    # Make prediction using the loaded model
    predicted_cost = medical_model.predict(client_data)

    # Return the prediction as JSON response
    return jsonify({'predicted_cost': predicted_cost[0]})

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)

