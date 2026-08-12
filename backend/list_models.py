from google import genai
from config import settings

client = genai.Client(api_key=settings.gemini_api_key)

print("Available Models:")
for model in client.models.list():
    if "flash" in model.name:
        print(model.name)
