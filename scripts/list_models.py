
import os
import requests
from dotenv import load_dotenv

load_dotenv("d:/agentic-rag/.env", override=True)
key = os.getenv("GEMINI_API_KEY")

def list_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    response = requests.get(url)
    if response.status_code == 200:
        models = response.json().get('models', [])
        for m in models:
            print(f"- {m['name']} ({m['displayName']})")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    list_models()
