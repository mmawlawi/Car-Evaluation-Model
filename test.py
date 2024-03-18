import requests
import json

# Define your input data
input_data = {
    "Brand": "Ssangyong",
    "Year": 2022,
    "Model": "Rexton",

    "UsedOrNew": "DEMO",
    "Transmission": "Automatic",
    "DriveType": "AWD",
    "FuelType": "Diesel",
    "FuelConsumption": 8.7,
    "Kilometres": 5595,
    "CylindersinEngine": 4,

    "Doors": 4,
    "Seats": 7,
    "EngineL": 2.2,
    "CarAge": 2,
    "State": "NSW"
}

# Send a POST request to the API endpoint
response = requests.post("http://127.0.0.1:5000/predict", json=input_data)

# Print the response
print(response.json())
