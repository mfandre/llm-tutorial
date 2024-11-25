import os
from openai import OpenAI
from dotenv import load_dotenv

# Carregando variáveis de ambiente a partir do arquivo .env
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")

# Configurar a chave da API
client = OpenAI(api_key=OPENAI_SERVICE_ACCOUNT_KEY)

# Subir o arquivo de treinamento
file_obj = client.files.create(file=open("finetune.jsonl", "rb"), purpose="fine-tune")

# Criar modelo fine-tuned
job = client.fine_tuning.jobs.create(training_file=file_obj.id, model="davinci-002")
print(job.fine_tuned_model)


# Quando o job finalizar você poderá acessar o nome do modelo criar para ser usado...
model_name = job.fine_tuned_model

completion = client.chat.completions.create(
    model=model_name, prompt="O que significa o código de erro E102?", max_tokens=100
)
print(completion.choices[0].message)
