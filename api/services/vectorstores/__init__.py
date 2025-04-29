from .retriever import build_retriever_tool


def load_tools():
    tools = [
        build_retriever_tool()
    ]

    return tools