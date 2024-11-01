# Agora vamos trabalhar com outputs estruturados, de forma simples, quero retornar uma entidade (objeto) preenchida ao invés de um string!
# Para isso iremos usar ell complex que nos ajudará nessa tarefa!

# importanto bibliotecas que usaremos
import os
from openai import OpenAI
import ell
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Carregando variáveis de ambiente
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")

# Iniciando a bibliteca ell com a flag verbose true. Isso fará as mensagens de debug muito mais intuitivas. Também estamos iniciando com o client da OpenAI. Aqu ivc pode usar outros clients como da Antrophic e outras LLMs
client = OpenAI(
    api_key=OPENAI_SERVICE_ACCOUNT_KEY,
)
ell.init(verbose=True, default_client=client)


# Vamos criar uma entidade que será usada para converter a resposta do ChatGPT. Pense que [prompt] -> [LLM] -> [Entidade]. O output não será mais string e sim a entidade que queremos! Isso é muito poderoso, podemos fazer muita coisa divertida com isso. Pode ser que eu traga mais exemplos práticos no futuro, me segue ai pra ficar ligado!!!
class DocumentTags(BaseModel):
    area: str = Field(description="The area of the document")
    subject: str = Field(description="The subject of the document")


# Perceba ques estamos colocando o parametro response_format que será usado para transformar a resposta.
@ell.complex(model="gpt-4o-mini", response_format=DocumentTags)
def generate_tags(document_content: str):
    """You are a document tagger generator. Given the content of a document, you need to return a structured tags about the document."""
    return f"Generate tags for the follow document {document_content}"


# Vamos chamar nossa função passando um texto sobre o grafeno
tag_message = generate_tags(
    """# Grafeno: Uma Revolução Tecnológica

O grafeno é um material composto por uma única camada de átomos de carbono dispostos em uma rede hexagonal. Ele é conhecido por suas propriedades excepcionais, como alta condutividade elétrica e térmica, além de ser extremamente leve e resistente. Descoberto em 2004, o grafeno tem potencial para revolucionar diversas indústrias, incluindo eletrônica, medicina e energia. Suas aplicações vão desde baterias mais eficientes até dispositivos médicos avançados, tornando-o um dos materiais mais promissores da atualidade."""
)
tags: DocumentTags = tag_message.parsed

# Vamos ver output:
print(f"Area: {tags.area}, Subject: {tags.subject}")

print("-----")


# Agora vamos testar nossa função passando um texto sobre o sol
tag_message = generate_tags(
    """# O Sol: A Estrela da Nossa Vida

O Sol é a estrela central do nosso sistema solar, responsável por fornecer luz e calor essenciais para a vida na Terra. Composto principalmente de hidrogênio e hélio, ele gera energia através da fusão nuclear em seu núcleo. Essa energia é liberada como luz e calor, sustentando a vida e influenciando o clima do nosso planeta. O Sol tem aproximadamente 4,6 bilhões de anos e está a cerca de 150 milhões de quilômetros da Terra. Sua importância é vital para a existência e o desenvolvimento de todas as formas de vida que conhecemos."""
)
tags: DocumentTags = tag_message.parsed

# Output:
print(f"Area: {tags.area}, Subject: {tags.subject}")

# Olha que foda! Agora podemos inserir texto e ter respostas estruturadas no formato que quisermos!
