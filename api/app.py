from fastapi import FastAPI
from controllers import chat, uploads



api = FastAPI(
    title="Agent with RAG",
    description="API for RAG chatbot",
    version="1.0.0"
)

api.include_router(chat.router, prefix="/api")
api.include_router(uploads.router, prefix="/api")