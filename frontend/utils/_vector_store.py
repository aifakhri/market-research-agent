from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def document_splitters():
    """
    """

    docs = [WebBaseLoader(url) for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=50
    )

    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

def vectorstore_retriever():
    """
    """

    vector_storage = Chroma.from_documents(
        documents=document_splitters(),
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings()
    )
    retriever = vector_storage.as_retriever()
    return retriever