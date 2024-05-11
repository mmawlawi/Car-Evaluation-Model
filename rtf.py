import pandas as pd
import json

df = pd.read_csv('Modified Australian Vehicle Prices.csv')
brand_model_df = df[['Brand', 'Model']]
unique_tuples = brand_model_df.drop_duplicates()

bmjd = dict()
for _, row in unique_tuples.iterrows():
    brand = row['Brand']
    model = row['Model']
    if brand in bmjd:
        bmjd[brand].add(model)
    else:
        bmjd[brand] = {model}

for brand, models in bmjd.items():
    bmjd[brand] = list(models)
 
type_df = df['BodyType']
unique_types = type_df.drop_duplicates()  

tjd = unique_types.tolist()
print(tjd)

fc_df = df['FuelConsumption']
fcjd = fc_df.drop_duplicates().tolist()


        
json_data = json.dumps({"Brand-Model":bmjd , "BodyType":tjd , "FuelConsumption":fcjd} , indent=4)
with open('data_labels.json', 'w') as json_file:
    json_file.write(json_data)