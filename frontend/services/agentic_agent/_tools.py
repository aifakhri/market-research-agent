from langchain.tools.retriever import create_retriever_tool

from utils import vectorstore_retriever


def retriever():
    """Search and return information about Lilian WEng blog post on LLM Agent"""
    retriever_tool = create_retriever_tool(
        vectorstore_retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian WEng blog post on LLM Agent"
    )
    return retriever_tool
