from pydantic import BaseModel, Field
from typing import List

class DeepResearchWithReasoning(BaseModel):
    """Relatório de pesquisa detalhado com raciocínio incluído."""
    question_summary: str = Field(description="Resumo conciso da pergunta.")
    research_steps: List[str] = Field(description="Queries de busca executadas.")
    key_findings: List[str] = Field(description="Fatos importantes encontrados.")
    synthesis: str = Field(description="Combinação das descobertas em uma visão coesa.")
    reasoning: str = Field(description="Descrição detalhada do raciocínio do agente passo a passo.")
    final_answer: str = Field(description="Resposta final baseada na síntese e raciocínio.")
