import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Carregando variáveis de ambiente a partir do arquivo .env
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")

# Configurar a chave da API
client = OpenAI(api_key=OPENAI_SERVICE_ACCOUNT_KEY)

# Lista de documentos
documents = [
    "A máquina precisa de manutenção preventiva mensal.",
    "O motor apresenta superaquecimento constante.",
    "O filtro hidráulico deve ser trocado após 500 horas de operação.",
]

# Gerar embeddings para os documentos
document_embeddings = [
    client.embeddings.create(input=doc, model="text-embedding-ada-002")["data"][0]["embedding"]
    for doc in documents
]

# Função para calcular similaridade (usando produto escalar)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Busca semântica
query = "Quando devo trocar o filtro hidráulico?"
query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]

# Encontrar o documento mais relevante
similarities = [
    cosine_similarity(query_embedding, doc_emb) for doc_emb in document_embeddings
]

# > b = [0, 5, 2, 3, 4, 5]
# > np.argmax(b)  # Only the first occurrence is returned.
# > 1
# argmax retorna o índice do maior valor de um vetor
most_similar_doc_index = np.argmax(similarities)

print("Documento mais relevante:", documents[most_similar_doc_index])
