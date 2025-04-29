from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage


from services.agents.graph import AgenticGraph
from models import (
    ChatRequest,
    ChatResponse,
)



router = APIRouter()
graph = AgenticGraph().build()


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatRequest):
    if not chat_message.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty"
        )

    inputs = {
        "messages": [HumanMessage(content=chat_message.question)]
    }

    responses = graph.invoke(inputs)

    if not responses.get("messages"):
        raise HTTPException(
            status_code=500,
            detail="No response generated"
        )

    response_text = responses["messages"][-1].content

    return ChatResponse(answer=response_text)
    