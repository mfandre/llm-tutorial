import faiss  # pip install faiss ou pip install faiss-cpu
import numpy as np

# Exemplo de embeddings (gerados por um modelo, como o da OpenAI)
embeddings = np.array(
    [
        [0.1, 0.2, 0.3],  # Embedding 1
        [0.4, 0.5, 0.6],  # Embedding 2
        [0.7, 0.8, 0.9],  # Embedding 3
    ]
).astype("float32")

# Criar um índice FAISS
dimension = embeddings.shape[1]  # Número de dimensões dos vetores

index = faiss.IndexFlatL2(dimension)  # Usando a métrica L2 (distância euclidiana)

index.add(embeddings)  # Adicionar embeddings ao índice

# Consultar o índice
query_vector = np.array([[0.15, 0.25, 0.35]]).astype("float32")  # Consulta

distances, indices = index.search(
    query_vector, k=2
)  # Retorna os 2 vetores mais próximos

print("Índices mais próximos:", indices)
print("Distâncias correspondentes:", distances)
