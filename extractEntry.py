import pandas as pd

# Assuming the CSV file is named 'your_file.csv' and located in the same directory as your script
csv_file_path = 'Modified Australian Vehicle Prices.csv'

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Convert the first row to a JSON string
first_entry_json = df.iloc[0].to_json()

# Define the path for the JSON file you want to create
json_file_path = 'first_entry.json'

# Write the JSON string to the file
with open(json_file_path, 'w') as file:
    file.write(first_entry_json)

print(f"First entry has been saved to {json_file_path}")
