# Agora vamos trabalhar para melhorar a legibilidade do nosso antigo código usando uma biblioteca chamada ell!
# Vamos tentar modularizar nosso código de forma simplificar o código gerado e tornar ele mais legível e interpretável

# importanto as bibliotecas que usaremos
import os
import ell
from dotenv import load_dotenv
from openai import OpenAI

# Carregando variáveis de ambiente
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")


# Iniciando a bibliteca ell com a flag verbose true. Isso fará as mensagens de debug muito mais intuitivas. Também estamos iniciando com o client da OpenAI. Aqu ivc pode usar outros clients como da Antrophic e outras LLMs
client = OpenAI(
    api_key=OPENAI_SERVICE_ACCOUNT_KEY,
)
ell.init(verbose=True, default_client=client)


# Aqui estou falando que minha função hello irá usar o modelo gpt-4o-mini e responderá o prompts utlizando no máx 100 tokens.
@ell.simple(model="gpt-4o-mini", max_tokens=100)
def hello(name: str):
    """You are a helpful assistant."""  # System prompt
    return f"Diga olá para {name}!"  # User prompt


# Ao inves de usar Python DocString como system roles nós podemos definir de forma explícita no nosso retorno.
@ell.simple(model="gpt-4o-mini")
def hello2(name: str):
    return [
        ell.system("Você é um assistente gentil e prestativo."),
        ell.user(f"Diga olá para {name}!"),
    ]


world = hello("Mundo")
print(world)

world = hello2("Mundo")
print(world)

# ell simple é muitop útil quando desejamos uma interação direta, sem pensar em histórico. Pergunta -> Resposta
# Veja como o nosso código ficou muito mais limpo e intuitívo. Não tivemos que nos preocupar em formatar nosso input nem como fazer parser do nosso output. Estamos trabalhando diretamente com strings de entrada e saída!
# Além de termos um código muito mais legível!
