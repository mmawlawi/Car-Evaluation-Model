from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
from pandas import DataFrame

# Load preprocessor and model
preprocessor = load('preprocessor.joblib')
model = load('pipeline_cat.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert JSON to DataFrame
    input_data = DataFrame([data])

    # Preprocess input data
    preprocessed_data = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(preprocessed_data)

    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)