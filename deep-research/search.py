import os
from dotenv import load_dotenv

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings


from dotenv import load_dotenv

from models import DeepResearchWithReasoning
from pretty_print import pretty_print_report

load_dotenv(override=True)


# --- CONFIGURAÇÃO E CARREGAMENTO DO ÍNDICE ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("ERRO: Chave OPENAI_API_KEY não encontrada no arquivo .env")
    exit()

ARQUIVO_INDICE_FAISS = "indice.faiss"
ARQUIVO_MAPA_DOCUMENTOS = "documentos.pkl"

model = OpenAIResponsesModel('o3-mini')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
    openai_reasoning_summary='detailed',
)
agent = Agent(model, model_settings=settings,output_type=DeepResearchWithReasoning,)

# Carrega o modelo (deve ser o mesmo usado na indexação)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Carrega o índice e os documentos
try:
    index = faiss.read_index(ARQUIVO_INDICE_FAISS)
    with open(ARQUIVO_MAPA_DOCUMENTOS, 'rb') as f:
        document_chunks = pickle.load(f)
except FileNotFoundError:
    print("ERRO: Arquivos de índice não encontrados. Execute 'index.py' primeiro.")
    exit()

# --- DEFINIR A FERRAMENTA DE PESQUISA LOCAL ---
@agent.tool
def local_vector_search(context: RunContext, query: str) -> str:
    """
    Busca informações em uma base de conhecimento local sobre pontos turísticos do Rio de Janeiro.
    Use para encontrar detalhes e relações entre Parque Lage, Jardim Botânico e Cristo Redentor.
    Retorna os trechos de texto mais relevantes para a consulta.
    """
    print(f"--- 🔎 Executando busca vetorial local para: '{query}' ---")
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    
    # Busca os 2 vizinhos mais próximos (k=2)
    distances, indices = index.search(query_vector, k=2)
    
    # Pega os textos correspondentes aos índices encontrados
    results = [document_chunks[i] for i in indices[0]]
    
    return "\n\n---\n\n".join(results)


# --- CONFIGURAR E EXECUTAR O PydanticAI ---
# Uma pergunta que requer a combinação de informações dos arquivos
query = """
Qual a relação entre o Parque Lage e o Jardim Botânico, 
e como o Cristo Redentor se encaixa na paisagem vista de ambos?
"""

print(f"🚀 Iniciando pesquisa LOCAL estruturada com Pydantic-AI...\n")

prompt=f"""
Você é um agente de pesquisa profunda com acesso a uma base local.
Sua tarefa é responder a pergunta a seguir, executando buscas locais conforme necessário.
Explique seu raciocínio passo a passo antes de responder.

Pergunta: {query}

IMPORTANTE:
- Use APENAS informações retornadas pela ferramenta de busca local.
- Documente claramente cada etapa da pesquisa no campo `research_steps`.
- Justifique suas conexões e conclusões no campo `reasoning`.
"""

report = agent.run_sync(prompt)
output = report.output

# --- 5. ANALISAR A SAÍDA ESTRUTURADA ---
print("\n✅ Pesquisa Concluída. Relatório Estruturado Gerado:\n")
pretty_print_report(output)
