from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from const import LLM_MODEL


def llm():
    """Calling chat model provider
    TODO: Add another chat model
    """

    return ChatOpenAI(model=LLM_MODEL)

def embeddings():
    """Calling embeddings from model provider
    TODO: Add another embedding from another chat model
    """

    return OpenAIEmbeddings()