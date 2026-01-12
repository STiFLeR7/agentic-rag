
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("d:/agentic-rag/.env")
api_key = os.getenv("GEMINI_API_KEY")

print(f"Key loaded: {api_key[:4]}... len={len(api_key)}")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

try:
    response = model.generate_content("Hello")
    print(f"Response: {response.text}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
