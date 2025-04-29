from fastapi import APIRouter, UploadFile, File, HTTPException
from services.vectorstore import load_vectorstore
from services.preprocessing.loader import load_doc_contents
from services.preprocessing.splitter import split_documents



router = APIRouter()
vectorstore = load_vectorstore()


@router.post("/upload")
async def upload_document(file: UploadFile=File(...)):
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Only '.txt' or '.md' files"
        )

    try:
        content = (await file.read()).decode("utf-8")

        documents = load_doc_contents(content, file.filename)
        splitted_documents = split_documents(documents=documents)

        vectorstore.upsert_documents(splitted_documents)

        return {"message": f"Uploaded and indexed is successfull"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")