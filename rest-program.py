import requests
import json

url = "http://localhost:11434/api/generate"
data = {
    "model": "medllama2",
    "prompt": "A 35-year-old man getting stomach ache at night, what could be the diagnosis?",
    "stream": True
}

response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        parsed = json.loads(line.decode('utf-8'))
        print(parsed.get("response", ""), end="", flush=True)
