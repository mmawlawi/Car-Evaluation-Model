from flask import Flask, request, jsonify
from joblib import load
from pandas import DataFrame

app = Flask(__name__)


preprocessor = load('preprocessor.joblib')
model = load('pipeline_cat.joblib')


EXPECTED_FIELDS = {
    "Brand": str,
    "Year": int,
    "Model": str,
    "Car/Suv": str,
    "UsedOrNew": str,
    "Transmission": str,
    "DriveType": str,
    "FuelType": str,
    "FuelConsumption": float,
    "Kilometres": int,
    "CylindersinEngine": int,
    "BodyType": str,
    "Doors": int,
    "Seats": int,
    "EngineL": float,
    "CarAge": int,
    "State": str
}

def validate_input_data(data):
  
    for field in EXPECTED_FIELDS:
        if field not in data:
            return False, f"Missing field: {field}"
    
    for field, data_type in EXPECTED_FIELDS.items():
        if not isinstance(data[field], data_type):
            return False, f"Field {field} should be of type {data_type.__name__}"
    
    return True, "Input data is valid"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    is_valid, message = validate_input_data(data)
    if not is_valid:
        return jsonify({'error': message}), 400

    try:
        input_data = DataFrame([data])

        # Ensure data types match expected types
        for col in input_data.columns:
            if col in EXPECTED_FIELDS:
                input_data[col] = input_data[col].astype(EXPECTED_FIELDS[col])

        # Predict
        prediction = model.predict(input_data)

        # Return prediction
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # Log and return the error if any step fails
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

