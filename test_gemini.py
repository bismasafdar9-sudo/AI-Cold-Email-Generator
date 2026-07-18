import os
from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv()

# Read API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found!")
    exit()

# Create client
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Hello"
    )

    print("\n✅ AI Connected Successfully!\n")
    print(response.text)

except Exception as e:
    print("\n❌ Error:")
    print(e)