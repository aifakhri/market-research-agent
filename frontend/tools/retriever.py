from typing import List
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader


from langchain.tools.retriever import create_retriever_tool

from ..const import QDRANT_ADDRESS, OPENAI_EMBEDDING, QDRANT_COLLECTION


URLS = [
    "https://lilianweng.github.io/lil-log/2023/06/23/agent.html",
    "https://lilianweng.github.io/lil-log/2023/06/15/prompt-engineering.html",
    "https://lilianweng.github.io/lil-log/2023/06/09/llm-attacks.html",
]



def load_documents() -> List[Document]:
    loader = WebBaseLoader(URLS)
    return loader

def split_documents(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def connect_to_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_ADDRESS)

def ensure_collection(client: QdrantClient, vector_size: int):
    collections = client.get_collections().collections
    if QDRANT_COLLECTION not in [c.name for c in collections]:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size, distance=Distance.COSINE
            )
        )

def create_vectorstore(
        splitted_docs: List[Document],
        client: QdrantClient,
        embeddings: OpenAIEmbeddings
    ) -> QdrantVectorStore:
    """
    """

    vectorstore = QdrantVectorStore(
        clients=client,
        documents=splitted_docs,
        embedding=embeddings
    )

    return vectorstore

def create_retriever_tool():
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING)
    vector_size = embeddings.embed_query("probe vector size")

    docs = load_documents()
    splitted_docs = split_documents(docs)

    client = connect_to_qdrant()
    ensure_collection(client=client, vector_size=vector_size)

    vectorstore = create_vectorstore(
        splitted_docs=splitted_docs, client=client, embeddings=embeddings
    )

    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        name="qdrant_retriever",
        description="Q&A based on Lilian Weng's blog articles"
    )

    return retriever_tool