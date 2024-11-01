import os

####################################
# carregando variaveis de ambiente #
####################################
from dotenv import load_dotenv

load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")

#############################################
# Vamos listar todos os modelos disponíveis #
#############################################
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_SERVICE_ACCOUNT_KEY,
)

print(client.models.list())

###################################################
# Vamos fazer uma chamada simples para o chat GPT #
###################################################
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "Você é um assistente gentil e prestativo."},
        {
            "role": "user",
            "content": "Fale: Hello World",
        },
    ],
    model="gpt-4o-mini",
)

print(chat_completion)
print("--------------")
print(chat_completion.choices[0].message.content)

# Pense que uma LLM é um função que recebe um texto de entrada (um prompt) e retorna um texto de saída (resposta)
# prompt -> llm -> resposta
# nesse caso demos duas instruções
# 1. Role System: Você é um assistente gentil e prestativo.
# 2. Rolse User: Fale: Isso é um TESTE!
# > System é usada para fornecer informações de configuração ou contexto que tenta direcionar o comportamento do modelo
# > User é usada como se fosse um input do usuário
# > Assistant é a resposta do modelo


# repare que vc pode fazer a mesma interação usando a REST API da OpenAI... porém a biblioteca facilita nossa interação!
# https://platform.openai.com/docs/api-reference/making-requests
