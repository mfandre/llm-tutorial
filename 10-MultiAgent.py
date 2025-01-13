# O objetivo desse código é mostrar um exemplo simples de multi agents. A idéia ter dois agentes, um que recebe um texto e resume. O outro agente é responsável por ler os resumos e tomar uma decisão.
# Toda a comunicação é feita por filas (nessa caso para exemplo estou usando um bliboteca baseada em sqlite, só pra manter isso simples!), tentei trazer essa complexidade para mostrar como isso funcionaria em uma ambiente distribuído.

# Deforma simplificada o processo segue o seguinte fluxo:
# Produtor (API, console...) gera um texto de entrada => Fila de Resumo => Agente especializado em Resumo pega o texto de entrada e Gera Resumo => Fila de decisão => Agente especializado em decisao gera uma decisao baseado no resumo => Fim

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import ell
import sys
import time

from litequeue import LiteQueue, Message


## Para executar usando OpenAI
# Carregando variáveis de ambiente a partir do arquivo .env
# load_dotenv(r".env")
# OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")

# # Configurar a chave da API
# client = OpenAI(api_key=OPENAI_SERVICE_ACCOUNT_KEY)


## Para executar unsando Ollama (gratuito) => veja como instalar aqui https://ollama.com/
MODEL = "llama3.1"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

ell.config.verbose = True
ell.config.register_model(MODEL, client)


# LLM responsável pelo Resumo, será usado no agente "Resumidor"
@ell.simple(model="llama3.1", client=client)
def analyse_and_summary(text: str):
    return [
        ell.system("You are an especialist agent that analyse text and sumarizes."),
        ell.user(f"Analyze the following data and provide a summary: {text}"),
    ]


# LLM responsável pela Decisão, será usado no agente "Decisor"
@ell.simple(model="llama3.1", client=client)
def take_action(summary: str):
    return [
        ell.system(
            "You are an especialist agent that makes decision based on summaries."
        ),
        ell.user(
            f"Based on the following summary: '{summary}', what is the best decision to make?"
        ),
    ]


if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)

    db_path = "queue.sqlite3"

    # O ideal aqui era que cada um desses trechos fossem separados em arquivos. Porém pra facilitar o tutorial, coloquei tudo em um único arquivo.

    # Trecho que representa uma aplicação produtra ("Producer")
    if len(args) == 0:
        print("--- Init start Producer ---")
        data = r"Data shows that sales increased by 20% in the last quarter due to a new marketing campaign."
        q1 = LiteQueue(db_path, queue_name="topic1")
        q1.put(data)
    # Trecho que representa nosso agente responsável pela sumarização
    elif args[0] == "summarizer":
        print("--- Init Summarizer ---")
        q1 = LiteQueue(db_path, queue_name="topic1")
        q2 = LiteQueue(db_path, queue_name="topic2")
        while True:
            if q1.qsize() == 0:
                continue
            data: Message = q1.pop()
            if data is None:
                continue
            summary = analyse_and_summary(str(data.data))
            q2.put(summary)
            time.sleep(0.1)
    # Trecho que representa nosso agente responsável pela decisão
    elif args[0] == "actioner":
        print("--- Init Actioner ---")
        q2 = LiteQueue(db_path, queue_name="topic2")
        while True:
            if q2.qsize() == 0:
                continue
            data: Message = q2.pop()
            if data is None:
                continue
            action = take_action(str(data.data))
            print(action)
            time.sleep(0.1)


# Para rodar o exemplo abra 3 terminais (consoles).
# No primeiro execute:
# > python 10-MultiAgent.py actioner
# No segundo execute:
# > python  10-MultiAgent.py summarizer
# No terceiro execute:
# > python 10-MultiAgent.py

# Sem uso de filas a lógica seria algo assim:
# # Fluxo de comunicação entre os agentes
# summary = agent1(data)
# decision = agent2(summary)
# print(f"Resumo do Agente 1: {summary}")
# print(f"Decisão do Agente 2: {decision}")
