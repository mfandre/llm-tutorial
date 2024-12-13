import os
from typing import List, Callable
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion
import json
from pprint import pp

# Carregando variáveis de ambiente
load_dotenv(r".env")
OPENAI_SERVICE_ACCOUNT_KEY = os.getenv("OPENAI_SERVICE_ACCOUNT_KEY")


# Iniciando a bibliteca ell com a flag verbose true. Isso fará as mensagens de debug muito mais intuitivas. Também estamos iniciando com o client da OpenAI. Aqu ivc pode usar outros clients como da Antrophic e outras LLMs
# client = OpenAI(
#     api_key=OPENAI_SERVICE_ACCOUNT_KEY,
# )
# ell.init(verbose=True, default_client=client)

## Para executar unsando Ollama (gratuito) => veja como instalar aqui https://ollama.com/
MODEL = "llama3.1"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Our FMEA mock Database
# FMEA source... Do not use it in production, it should be get from a better source... ITS A EXAMPLE
FMEAs = {
    "turbine": {
        "blade wear": "failure mode: Blade wear\n\n failure effect: Reduced efficiency\n\n failure cause: Regular inspections and replacement of worn blades\n\n action: Regular inspections and replacement of worn blades\n---\n\n",
        "excessive vibration": "failure mode: Excessive Vibration\n\n failure effect: Structural damage\n\n failure cause: Imbalance\n\n action: Periodic balancing and vibration monitoring\n\n\n---",
    },
    "condenser": {
        "leakage": "failure mode: Leakage\n\n failure effect: Loss of efficiency\n\n failure cause: Corrosion or physical damage\n\n action: Regular inspections and repair of leaks\n---\n\n",
        "clog": "failure mode: Obstruction\n\n failure effect: Reduced heat exchange capacity\n\n failure cause: Accumulation of debris\n\n action: Periodic cleaning and installation of filters\n\n\n---\n\n",
    },
}


def get_all_equipments_in_FMEA():
    return str(list(FMEAs.keys()))


def get_all_failure_modes_in_FMEA_by_equipment_name(equipment_name: str):
    """Get an option list of failure modes from equiment name available in FMEA. It should be used to give options to the user select the failure mode available for the given equipment."""  # Tool description
    equipment_name = equipment_name.lower()
    # print("Get equiment FMEA failure modes", equipment_name)

    if equipment_name.lower() not in FMEAs:
        return f"No FMEA document for the equipment '{equipment_name}' found."

    return str(list(FMEAs[equipment_name.lower()].keys()))


def get_solution_in_FMEA_by_equipment_name_and_failure_mode(
    equipment_name: str, equipment_failure_mode: str
):
    """Get FMEA for equiment based on equipment name and the described failure mode."""

    equipment_name = equipment_name.lower()
    equipment_failure_mode = equipment_failure_mode.lower()

    print("Get equiment FMEA", equipment_name)

    # this should get from your database/files (FMEA source)
    if equipment_name.lower() not in FMEAs:
        return f"No FMEA found for the equipment '{equipment_name}' and failure mode '{equipment_failure_mode}'."

    if equipment_failure_mode.lower() not in FMEAs[equipment_name.lower()]:
        return f"No FMEA found for the equipment '{equipment_name}' and failure mode '{equipment_failure_mode}'."

    FMEA = FMEAs[equipment_name.lower()][equipment_failure_mode.lower()]

    return FMEA


def convert_user_fail_description_2_fmea_failure_modes(
    equipment_name: str, fail_description: str
):
    available_failure_modes = str(list(FMEAs[equipment_name.lower()].keys()))

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant, and you must convert a description of a device failure into the possible categories: {available_failure_modes}. Your answer must contain only a single category written exactly as the categories provided above. Your answer must not contain any special characters or accents!",
        },
        {
            "role": "user",
            "content": f"Convert the following user description {fail_description}!",
        },
    ]

    completion = get_completion(messages)
    return (
        "equipment failure mode identified in the FMEA database (equipment_failure_mode) =>"
        + completion.choices[0].message.content
    )


def get_system_prompt() -> str:
    return "You are a BOT that is an expert in searching for information in an FMEA database. You have access to several functions to diagnose the equipment and guide the mechanical engineers in solving the problem. Typically, the user will provide the equipment and a description of the failure mode. If at the end of the conversation no FMEA is found, you MUST respond 'Sorry, I can't find an FMEA for that'! You must not suggest possibilities and should base your answer only on what you have described in the FMEA database!"


def get_llm_functions() -> List[dict[str, str]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_all_equipments_in_FMEA",
                "description": "Returns all equipment names that have FMEA registered in the FMEA database.",
                "parameters": None,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_all_failure_modes_in_FMEA_by_equipment_name",
                "description": "Returns a list of failure modes existing in the FMEA database given an equipment name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equipment_name": {
                            "type": "string",
                            # PART OF THE PROMPT! Careful what you write here
                            "description": "Equipment name to return failure modes.",
                        }
                    },
                    "required": ["equipment_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_user_fail_description_2_fmea_failure_modes",
                "description": "Given the failure mode description written by the user, identify it in the corresponding failure mode in the FMEA database for the reported equipment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equipment_name": {
                            "type": "string",
                            # PART OF THE PROMPT! Careful what you write here
                            "description": "Equipment name provided by the user.",
                        },
                        "fail_description": {
                            "type": "string",
                            # PART OF THE PROMPT! Careful what you write here
                            "description": "Description of the failure mode reported by the user",
                        },
                    },
                    "required": ["equipment_name", "fail_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_solution_in_FMEA_by_equipment_name_and_failure_mode",
                "description": "Searches for solutions in the FMEA database given the equipment name and failure mode. This function returns diagnostic information and possible solutions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equipment_name": {
                            "type": "string",
                            # PART OF THE PROMPT! Careful what you write here
                            "description": "Equipment name",
                        },
                        "equipment_failure_mode": {
                            "type": "string",
                            # PART OF THE PROMPT! Careful what you write here
                            "description": "Failure mode of existing equipment in FMEA base",
                        },
                    },
                    "required": ["equipment_name", "equipment_failure_mode"],
                },
            },
        },
    ]


# Get completion from the messages. It is a helper function which wraps a call to LLM
def get_completion(messages: List[dict[str, str]], tools=None) -> ...:
    res = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.01,
        max_tokens=150,
        # seed=3678,
    )
    return res


def controller(
    functions: dict[str, Callable] = None, chat_history: List = []
) -> ChatCompletion:
    # functions
    llm_functions = get_llm_functions()

    # Generate LLM response with messages and functions
    # Prompt is already in the messages stored as system role.
    completion = get_completion(
        messages=chat_history,
        tools=llm_functions,
    )

    # Verify if completion has `tool_calls` which is
    # List[ChatCompletionMessageToolCall] or None
    is_tool_call = completion.choices[0].message.tool_calls
    if is_tool_call:
        tool_call = completion.choices[0].message.tool_calls[0]

        # We need call ID, and function out of it. ID has to be send back to LLM later
        fn = functions[tool_call.function.name]
        args = json.loads(tool_call.function.arguments)

        # Call the function
        res = fn(**args)

        # Add messages. Both of them are essential for the correct call.
        # Add assistant's response message
        chat_history.append(completion.choices[0].message)
        # Add function calling result
        chat_history.append(dict(role="tool", tool_call_id=tool_call.id, content=res))

        # Run completion again to get the answer
        tool_completion = get_completion(messages=chat_history)

        pp(chat_history)

        # Return response which was generated with help of function calling
        return tool_completion.choices[0].message.content

    # Return response without function calling
    return completion.choices[0].message.content


if __name__ == "__main__":
    available_functions = dict(
        get_solution_in_FMEA_by_equipment_name_and_failure_mode=get_solution_in_FMEA_by_equipment_name_and_failure_mode,
        get_all_failure_modes_in_FMEA_by_equipment_name=get_all_failure_modes_in_FMEA_by_equipment_name,
        get_all_equipments_in_FMEA=get_all_equipments_in_FMEA,
        convert_user_fail_description_2_fmea_failure_modes=convert_user_fail_description_2_fmea_failure_modes,
    )

    user_messages = [
        "Hello!",
        "My condenser has stopped working. It is not exchanging heat as it should because it is clogged.",
        "I would like to search the FMEA database for solutions to what I described above.",
    ]
    # Set up first messages
    prompt = get_system_prompt()
    messages = [
        {"role": "system", "content": prompt},
    ]
    for user_message in user_messages:
        messages.append({"role": "user", "content": user_message})
        # pp(user_message)
        completion = controller(functions=available_functions, chat_history=messages)
        # pp(completion)
        messages.append({"role": "assistant", "content": completion})

        print("------------")
        pp(messages)
