import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")
model_name = "ft:gpt-3.5-turbo-0125:iamferraz::AXdYDa3O"
client = OpenAI(api_key=OPENAI_SERVICE_ACCOUNT_KEY)

# Ejemplo de uso
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")


# Para deletar finetuned:
client.models.delete(model_name)
