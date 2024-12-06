## Vamos implementar um RAG no contexto de manutenção preventida. O objetivo é o usuário entrar com uma perguntar e o sistema responder com base na base de conhecimento de manutenção da empresa.

## Esse RAG é bem simples pois utiliza apenas dados locais, a ideia é mostrar o conceito por trás da arquitetura RAG. Esse código não é ideal para um ambiente produtivo!

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import ell
import faiss  # pip install faiss ou pip install faiss-cpu


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


def calculate_embbeding(text: str) -> list[float]:
    """
    Converte um texto em um vetor. Esses vetores podem ser usados para serem armazenados em uma base de vetores por exemplo.
    """
    return (
        # client.embeddings.create(input=text, model="text-embedding-ada-002")
        client.embeddings.create(input=text, model="all-minilm")
        .data[0]
        .embedding
    )


def create_kb(corpora: list[str]) -> faiss.IndexFlatL2:
    """
    Função responsável por criar base de conhecimento. Calcular embedding e inserir no banco
    """

    # Gerar embeddings para os documentos
    document_embeddings = [calculate_embbeding(doc) for doc in corpora]

    embeddings = np.array(document_embeddings).astype("float32")

    # Criar um índice FAISS
    dimension = embeddings.shape[1]  # Número de dimensões dos vetores

    index = faiss.IndexFlatL2(dimension)  # Usando a métrica L2 (distância euclidiana)

    index.add(embeddings)  # Adicionar embeddings ao índice

    return index


def search_in_kb(query: str, corpora: list[str], kb: faiss.IndexFlatL2) -> list[str]:
    """
    Dado um texto, busca na base de conhecimento o dois documentos mais próximos
    """
    query_vector = np.array([calculate_embbeding(query)]).astype("float32")  # Consulta

    distances, indices = kb.search(
        query_vector, k=2
    )  # Retorna os 2 vetores mais próximos

    result = []

    for i in indices[0]:
        result.append(corpora[i])

    return result


# @ell.simple(model="gpt-4o-mini", client=client)
@ell.simple(model="llama3.1", client=client)
def generate_final_response(similar_docs: list[str], query: str):
    context = ""

    for doc in similar_docs:
        context += doc + "\n\n --- \n\n"

    return [
        ell.system("Baseada no contexto responda a pergunta fornecida."),
        ell.user(
            f"""Contexto:
{context}
Fim do Contexto
###
{query}"""
        ),
    ]


# Base de conhecimento de manutenção preventiva da empresa
corpora = [
    "O código de erro E102 indica superaquecimento do motor. Verifique o nível de óleo e o sistema de resfriamento.",
    "Recomenda-se trocar o óleo do gerador a cada 250 horas de operação ou a cada 6 meses, o que ocorrer primeiro. Verifique também o filtro de óleo a cada troca.",
    "Como calibrar uma balança industrial? Para calibrar a balança, coloque pesos padrão sobre a plataforma e ajuste o visor para refletir o peso correto. Repita o processo em diferentes pontos de carga.",
    "O que deve ser inspecionado em uma empilhadeira antes de usá-la? Verifique o nível de combustível ou carga da bateria, o funcionamento dos pneus, a integridade dos freios, a carga máxima permitida e os controles hidráulicos.",
]

# Input do usuário
query = "O que significa o erro E102?"

## Com base no post https://andredemattosferraz.substack.com/p/desvendando-a-genai-parte-11 estes seriam os três pilares do RAG
# Pilar 1 => Processaento do arquivo + Construção da base de conhecmento
kb = create_kb(corpora)

# Pilar 2 => Recuperação de Informação (Query)
similar_docs = search_in_kb(query, corpora, kb)

# Pilar 3 => Core
response = generate_final_response(similar_docs, query)
print(response)

response = generate_final_response([], query)  # avaliando resposta sem contexto
print(response)

## É possível notar que a resposta gerada sem um contexto não faz sentido para o usuário.
