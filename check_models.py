
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=api_key)

print("\nAVAILABLE GEMINI MODELS\n")

for model in client.models.list():
    actions = model.supported_actions or []

    if "generateContent" in actions:
        print(model.name)
