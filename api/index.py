# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')

#uest.for Load the trained model and the scaler (from pipeline) and the lambda_
model = joblib.load(os.path.join("../model", "best_model.joblib"))
scaler = joblib.load(os.path.join("../model", "StandardScaler.joblib"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs from the HTML form
        data = [
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['living_area']),
            float(request.form['lot_area']),
            float(request.form['floors']),
            int(request.form['waterfront']),
            int(request.form['views']),
            int(request.form['condition']),
            int(request.form['grade']),
            float(request.form['basement_area']),
            int(request.form['built_year']),
            int(request.form['renovation_year']),
            int(request.form['postal_code']),
            float(request.form['lattitude']),
            float(request.form['longitude']),
            float(request.form['lot_area_renov']),
            int(request.form['schools_nearby']),
            float(request.form['airport_distance'])
        ]

        # Define input column names corresponding to model features
        columns = [
            'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
            'number of floors', 'waterfront present', 'number of views',
            'condition of the house', 'grade of the house', 'Area of the basement', 'built_year',
            'renovation_year', 'Postal Code', 'Lattitude', 'Longitude', 'lot_area_renov',
            'Number of schools nearby', 'Distance from the airport'
        ]

        # Convert input into DataFrame
        df = pd.DataFrame([data], columns=columns)

        # --- Feature Engineering ---
        current_year = pd.to_datetime('today').year

        # Create 'Age' and 'RenovatedOrNot' features
        df['Age'] = current_year - df['built_year']
        df['RenovatedOrNot'] = np.where(df['renovation_year'] > 0, 1, 0)

        # Drop raw year columns as they are no longer needed
        df.drop(columns=['built_year', 'renovation_year'], inplace=True)

        # Final feature set that matches training
        final_features = [
            'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
            'number of floors', 'waterfront present', 'number of views',
            'condition of the house', 'grade of the house', 'Area of the basement', 'Age',
            'RenovatedOrNot', 'Postal Code', 'Lattitude', 'Longitude', 'lot_area_renov',
            'Number of schools nearby', 'Distance from the airport'
        ]

        # Apply scaler to input
        input_data = df[final_features]
        scaled_input = scaler.transform(input_data)

        # Reversing the cube root prediction        
        prediction_cbrt = model.predict(scaled_input)[0]
        prediction = prediction_cbrt ** 3 

        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        # Handle any errors gracefully
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=False)


