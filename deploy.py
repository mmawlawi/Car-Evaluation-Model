from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

model = load('pipeline_cat.joblib')

EXPECTED_FIELDS = {
    "Brand": "object",
    "Year": "float64",
    "Model": "object",
    "Car/Suv": "object",
    "UsedOrNew": "object",
    "Transmission": "object",
    "DriveType": "object",
    "FuelType": "object",
    "FuelConsumption": "float64",
    "Kilometres": "float64",
    "CylindersinEngine": "float64",
    "BodyType": "object",
    "Doors": "float64",
    "Seats": "float64",
    "EngineL": "float64",
    "CarAge": "float64",
    "State": "object"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Initialize DataFrame with expected types
    input_data = pd.DataFrame({field: pd.Series(dtype=typ) for field, typ in EXPECTED_FIELDS.items()})
    
    missing_fields = []
    for field, dtype in EXPECTED_FIELDS.items():
        if field in data:
            input_data.at[0, field] = data[field]
        else:
            missing_fields.append(field)

    try:
        # Predict
        prediction = model.predict(input_data)
        
        # Construct response including missing fields information
        response = {
            'prediction': prediction.tolist(),
            'missing_fields': missing_fields
        }
        
        return jsonify(response)
    except Exception as e:
        # Log and return the error if any step fails
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
