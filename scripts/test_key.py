
import os
import requests
from dotenv import load_dotenv

load_dotenv("d:/agentic-rag/.env", override=True)
key = os.getenv("GEMINI_API_KEY")

def test_key():
    print(f"Testing key: {key[:10]}...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": "Hello, are you there?"}]}]
    }
    response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_key()
