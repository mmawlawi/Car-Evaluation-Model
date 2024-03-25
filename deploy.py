from flask import Flask, request, jsonify, send_file
from io import BytesIO
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt

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
    
def get_aggregated_feature_importances(feature_names, feature_importances):
    original_categorical_features = ['Brand', 'Model', 'Transmission', 'DriveType', 'FuelType', 'Car/Suv', 'UsedOrNew', 'BodyType', 'State']
    numerical_features = ['FuelConsumption', 'Kilometres', 'CylindersinEngine', 'Doors', 'Seats', 'EngineL', 'CarAge']

    aggregated_importances = {feature: 0 for feature in original_categorical_features + numerical_features}

    for feature, importance in zip(feature_names, feature_importances):
        for original_feature in original_categorical_features:
            if feature.startswith(original_feature):
                aggregated_importances[original_feature] += importance
                break
        if feature in numerical_features:
            aggregated_importances[feature] = importance

    return aggregated_importances

def get_feature_names(column_transformer):
    """Get feature names from all transformers."""
    feature_names = []
    
    # Loop through each transformer in the column transformer
    for transformer in column_transformer.transformers_:
        transformer_name, transformer_obj, columns = transformer
        
        if transformer_name != 'remainder':
            if hasattr(transformer_obj, 'get_feature_names_out'):
                # If it's a transformer that changes the number of features
                names = transformer_obj.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                # If it's a transformer that does not change the number of features
                feature_names.extend(columns)
                
    return feature_names

def get_feature_importance_data():
    """
    This function encapsulates the logic to fetch and process feature importance data.
    It returns a dictionary with feature names as keys and their importances as values.
    """
    try:
        # Extract the CatBoost model from the pipeline
        catboost_model = model.named_steps['model']
        preprocessor = model.named_steps['preprocessor']

        # Get feature importances
        feature_importances = catboost_model.get_feature_importance()
        
        feature_names = get_feature_names(preprocessor)

        # Assuming feature_names is accessible; otherwise, it should be generated as previously discussed
        aggregated_importances = get_aggregated_feature_importances(feature_names, feature_importances)
        
        return aggregated_importances  # Return the dictionary directly
    except Exception as e:
        return {'error': str(e)}

@app.route('/aggregated-feature-importance', methods=['GET'])
def aggregated_feature_importance():
    data = get_feature_importance_data()
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
    else:
        return jsonify(data)

@app.route('/feature-importance-graph', methods=['GET'])
def feature_importance_graph():
    feature_importance = get_feature_importance_data()
    if 'error' in feature_importance:
        return jsonify({'error': feature_importance['error']}), 500
    
    # Create a bar graph
    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance.keys(), feature_importance.values(), color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

    # Save the plot to a BytesIO object and return it
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
