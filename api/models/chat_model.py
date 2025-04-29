from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from const import (
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    OPENROUTER_MODEL
)


def llm():
    """Calling chat model provider
    TODO: Add another chat model
    """

    return ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        model=OPENROUTER_MODEL,
    )