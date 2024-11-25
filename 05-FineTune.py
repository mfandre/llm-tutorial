import os
from openai import OpenAI
from dotenv import load_dotenv

# Carregando variáveis de ambiente a partir do arquivo .env
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")

# Configurar a chave da API
client = OpenAI(api_key=OPENAI_SERVICE_ACCOUNT_KEY)


client.files.create(
    file=open("finetune.jsonl", "rb"),
    purpose="fine-tune"
)

