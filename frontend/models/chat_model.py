from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from const import (
    OPEN_ROUTER_API_KEY,
    OPEN_ROUTER_URL,
    OPEN_ROUTER_MODEL,
    OPENAI_EMBEDDING
)



def llm():
    """Calling chat model provider
    TODO: Add another chat model
    """

    return ChatOpenAI(
        openai_api_key=OPEN_ROUTER_API_KEY,
        openai_api_base=OPEN_ROUTER_URL,
        model_name=OPEN_ROUTER_MODEL,
    )

def embeddings():
    """Calling embeddings from model provider
    TODO: Add another embedding from another chat model
    """

    return OpenAIEmbeddings(model=OPENAI_EMBEDDING)