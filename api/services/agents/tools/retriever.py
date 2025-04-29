from services.vectorstore import load_vectorstore
from langchain.tools.retriever import create_retriever_tool



def load_retriever_tool():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    return create_retriever_tool(
        retriever,
        "qdrant_retriever",
        "Q&A retriever"
    )