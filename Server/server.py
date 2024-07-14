import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), '..', 'Model', 'Medical_model_gradientR.joblib')

medical_model = joblib.load(model_path)

@app.route('/')
def LandingPage():
    return render_template('MediPredict.html')

@app.route('/predict_page')
def predict_page():
    return render_template('PredictPage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from POST request
        age = int(request.form['age'])
        sex = 1 if request.form['sex'].lower() == 'female' else 0
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoking_status'].lower() == 'yes' else 0
        region = request.form['region'].lower()

        # Encoding region
        region_mapping = {'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4}
        region_encoded = region_mapping.get(region, 0)

        client_data = np.array([[age, sex, bmi, children, smoker, region_encoded]])

        # Make prediction using the loaded model
        predicted_cost = medical_model.predict(client_data)[0]

        # Return the prediction as a new template
        return render_template('PredictionResult.html', predicted_cost=predicted_cost)
    
    except KeyError as e:
        return f"Missing form field: {e}", 400
    except ValueError as e:
        return f"Value error: {e}", 400

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
