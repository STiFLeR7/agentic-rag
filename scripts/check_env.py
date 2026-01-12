
import os
from dotenv import load_dotenv

env_path = "d:/agentic-rag/.env"
print(f"Loading .env from {env_path}")
load_dotenv(env_path)

key = os.getenv("GEMINI_API_KEY")

if key:
    print(f"GEMINI_API_KEY found. Length: {len(key)}")
    print(f"Starts with: {key[:4]}...")
else:
    print("GEMINI_API_KEY NOT found.")
