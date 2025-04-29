from services.agents.tools.retriever import load_retriever_tool



def load_tools():
    return [
        load_retriever_tool()
    ]