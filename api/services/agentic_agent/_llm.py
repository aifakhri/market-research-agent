from langchain_openai import ChatOpenAI

from const import (
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    OPENROUTER_MODEL
)


def chat_model():
    """Calling chat model provider
    TODO: Add another chat model
    """

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
    )