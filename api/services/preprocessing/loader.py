from langchain_core.documents import Document



def load_doc_contents(content: str, filename: str) -> list[Document]:
    return[Document(page_content=content, metadata={"source": filename})]