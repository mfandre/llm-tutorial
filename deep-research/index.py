import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

from models import DeepResearchWithReasoning

# --- 1. CONFIGURAÇÃO ---
DIRETORIO_DOCUMENTOS = "corpora"
ARQUIVO_INDICE_FAISS = "indice.faiss"
ARQUIVO_MAPA_DOCUMENTOS = "documentos.pkl"

# Modelo para transformar texto em vetores (executa localmente)
print("Carregando o modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Modelo carregado.")

# --- 2. LER DOCUMENTOS ---
document_chunks = []
for filename in os.listdir(DIRETORIO_DOCUMENTOS):
    if filename.endswith(".txt"):
        filepath = os.path.join(DIRETORIO_DOCUMENTOS, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            document_chunks.append(f.read())

if not document_chunks:
    print("Nenhum documento encontrado. Verifique o diretório e os arquivos.")
    exit()

print(f"Encontrados {len(document_chunks)} documentos para indexar.")

# --- 3. GERAR EMBEDDINGS ---
print("Gerando embeddings (vetores) para os documentos...")
embeddings = model.encode(document_chunks, convert_to_tensor=False)
embeddings = np.array(embeddings).astype('float32')
print("Embeddings gerados com sucesso.")

# --- 4. CRIAR E SALVAR O ÍNDICE FAISS ---
dimensao_vetor = embeddings.shape[1]
index = faiss.IndexFlatL2(dimensao_vetor)  # Usando distância L2 (Euclidiana)
index.add(embeddings)

print(f"Índice FAISS criado com {index.ntotal} vetores.")

# Salva o índice no disco
faiss.write_index(index, ARQUIVO_INDICE_FAISS)
print(f"Índice salvo em '{ARQUIVO_INDICE_FAISS}'")

# Salva a lista de textos originais para mapeamento posterior
with open(ARQUIVO_MAPA_DOCUMENTOS, 'wb') as f:
    pickle.dump(document_chunks, f)
print(f"Mapeamento de documentos salvo em '{ARQUIVO_MAPA_DOCUMENTOS}'")