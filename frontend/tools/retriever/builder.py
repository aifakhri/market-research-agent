from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from langchain.tools.retriever import create_retriever_tool

from models import embeddings
from ...const import QDRANT_ADDRESS


URLS = [
    "https://lilianweng.github.io/lil-log/2023/06/23/agent.html",
    "https://lilianweng.github.io/lil-log/2023/06/15/prompt-engineering.html",
    "https://lilianweng.github.io/lil-log/2023/06/09/llm-attacks.html",
]


class RetrieverToolBuilder:
    def __init__(self, collection_name):
        self.embeddings = embeddings()
        self.collection_name = collection_name
        self.client = QdrantClient(url=QDRANT_ADDRESS)
        self.vector_size = len(self.embeddings.embed_query("probe dimension"))
        self.vectorstore = None

    def load_documents(self):
        """Function to load document"""

        # NOTE: Document is based on langgraph
        # TODO: Change when our documents are needed

        doc_loader = WebBaseLoader(URLS)
        return doc_loader

    def ensure_collection(self):
        collections = self.client.get_collection().collections
        if self.collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vector_configs=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE)
            )

    def build(self):
        self.ensure_collection()

        # Prepare document
        docs = self.load_documents()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=200
        )

        splitted_docs = splitter.split_documents(docs)

        self.vectorstore = QdrantVectorStore(
            clients=self.client,
            documents=splitted_docs,
            embedding=self.embeddings
        )

        retriever = self.vectorstore.as_retriever()

        return create_retriever_tool(
            retriever,
            name="qdrant_retriever",
            description="Q&A from Lilian Weng's Blog"
        )