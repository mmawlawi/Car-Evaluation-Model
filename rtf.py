import pandas as pd
import json

df = pd.read_csv('Australian Vehicle Prices.csv')
brand_model_df = df[['Brand', 'Model']]
unique_tuples = brand_model_df.drop_duplicates()
print(unique_tuples)

for_json = dict()
for _, row in unique_tuples.iterrows():
    brand = row['Brand']
    model = row['Model']
    if brand in for_json:
        for_json[brand].add(model)
    else:
        for_json[brand] = {model}

for brand, models in for_json.items():
    for_json[brand] = list(models)
     
json_data = json.dumps(for_json , indent=4)
with open('data_labels.json', 'w') as json_file:
    json_file.write(json_data)