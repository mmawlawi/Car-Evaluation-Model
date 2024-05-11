import requests
url = "https://8758e14f52be70b810911695f8d5975e.serveo.net/feature-importance"
response = requests.get(url)
print(response.json())

# docker run -p 5000:5000 flask-app
# ssh -R Car-Eval-Model:80:localhost:5000 serveo.net