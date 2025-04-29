from langchain.text_splitter import RecursiveCharacterTextSplitter



def split_documents(documents: list, chunk_size=700, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents=documents)