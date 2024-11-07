import os

# Vamos brincar com a LLM GPT utilizando a biblioteca python da OpenAI. O objetivo é fazer uma interação simples com o ChatGPT para entender como iniciar nesse mundo de desenvolvimendo usando GenAI.

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

# > $ Hello World! Como posso ajudá-lo hoje?

# Pense que uma LLM é um função que recebe um texto de entrada (um prompt) e retorna um texto de saída (resposta)
# prompt -> llm -> resposta
# nesse caso demos duas instruções
# 1. Role System: Você é um assistente gentil e prestativo.
# 2. Rolse User: Fale: Isso é um TESTE!
# > System é usada para fornecer informações de configuração ou contexto que tenta direcionar o comportamento do modelo
# > User é usada como se fosse um input do usuário
# > Assistant é a resposta do modelo
# Por isso é importante manter o atributo "role" preenchido corretamente. Esse atributo define o papel de cada mensagem na interação com o chat, garantindo que o sequenciamento da conversa seja mantido de forma lógica e coerente.
# Repare que o parametro messages é um array, ele serve como "memória/histórico" da sua interação com o GPT. O GPT irá responder de acordo com as informações desse array.
# É importante lembrar que o histórico será contabilizado como tokens de entrada e consequentemente o custo irá aumentar de acordo com o tamanho do array messages!


# repare que vc pode fazer a mesma interação usando a REST API da OpenAI... porém a biblioteca facilita nossa interação!
# https://platform.openai.com/docs/api-reference/making-requests
