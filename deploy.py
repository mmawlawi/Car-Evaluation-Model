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
    

    input_data = DataFrame([data])

    # run python deploy.py and test.py you'll get success!!!. Flask works fine, the problem is from the preprocessor
    return jsonify({'success': message, 'input_data': input_data.to_dict(orient='records')})

    # # Preprocess input data ------ @Maher if you run this one
    # preprocessed_data = preprocessor.transform(input_data)

    # # Predict
    # prediction = model.predict(preprocessed_data)

    # # Return prediction
    # return jsonify({'prediction': prediction.tolist()})

    # @Maher the problem ---->>>>
    # return jsonify({'success': message, 'preprocessed_data': preprocessed_data.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True)
