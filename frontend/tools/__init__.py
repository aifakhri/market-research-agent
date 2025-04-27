from retriever import create_retriever_tool


def load_tools():
    tools = [
        create_retriever_tool()
    ]

    return tools