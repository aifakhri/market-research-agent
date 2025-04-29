from uuid import uuid4
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


from langchain.tools.retriever import create_retriever_tool

from const import (
    QDRANT_ADDRESS,
    QDRANT_COLLECTION,
    QDRANT_ADDRESS,
    QDRANT_API_KEY,
    OPENAI_EMBEDDING
)


URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/"
]


def load_documents() -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in URLS]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list

def split_documents(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def connect_to_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_ADDRESS, api_key=QDRANT_API_KEY)

def insert_documents(vectorstore: QdrantVectorStore, docs: list):
    doc_ids = [str(uuid4()) for _ in range(len(docs))]
    vectorstore.add_documents(documents=docs, ids=doc_ids)

def create_doc_collection(client: QdrantClient, vector_size: int):
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size, distance=Distance.COSINE
        )
    )

def create_vectorstore(
        splitted_docs: List[Document],
        client: QdrantClient,
        embeddings: OpenAIEmbeddings,
        vector_size: int
    ) -> QdrantVectorStore:
    """
    """

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings
    )

    if not client.collection_exists(QDRANT_COLLECTION):
        create_doc_collection(client=client, vector_size=vector_size)

    if client.count(collection_name=QDRANT_COLLECTION).count == 0:
        insert_documents(vectorstore=vectorstore, docs=splitted_docs)

    return vectorstore

def build_retriever_tool():
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING)
    vector_size = len(embeddings.embed_query("probe vector size"))

    loaded_docs = load_documents()
    splitted_docs = split_documents(loaded_docs)

    client = connect_to_qdrant()

    vectorstore = create_vectorstore(
        splitted_docs=splitted_docs,
        client=client,
        embeddings=embeddings,
        vector_size=vector_size
    )

    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "qdrant_retriever",
        "Q&A based on Lilian Weng's blog articles"
    )

    return retriever_tool