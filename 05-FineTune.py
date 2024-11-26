import os
from openai import OpenAI
from dotenv import load_dotenv

# Carregando variáveis de ambiente a partir do arquivo .env
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")

# Configurar a chave da API
client = OpenAI(api_key=OPENAI_SERVICE_ACCOUNT_KEY)

# # Subir o arquivo de treinamento
# file_obj = client.files.create(file=open("finetune.jsonl", "rb"), purpose="fine-tune")

# # Criar modelo fine-tuned
# job = client.fine_tuning.jobs.create(
#     training_file=file_obj.id,
#     # model="gpt-4o-mini-2024-07-18"
#     model="gpt-3.5-turbo-0125",
# )
# print(job.fine_tuned_model)

# O job demora uns 10min pra executar, é preciso acompanhar a evolução pelo dashboard: https://platform.openai.com/
# Quando o job finalizar você poderá acessar o nome do modelo criado para ser usado posteriormente
model_name = "ft:gpt-3.5-turbo-0125:iamferraz::AXdYDa3O"

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": "Vocé um chatbot amigável que ajuda com respostas sobre manutenção preventiva.",
        },
        {
            "role": "user",
            "content": "O que significa o código de erro E102?",
        },
    ],
    max_tokens=100,
)

print(completion.choices[0].message)
