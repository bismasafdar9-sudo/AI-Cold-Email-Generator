import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    models = client.models.list()

    print("\n✅ Available Models:\n")

    for model in models:
        print(model.name)

except Exception as e:
    print("❌ Error:")
    print(e)