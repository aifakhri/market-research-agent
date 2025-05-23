from pydantic import BaseModel, Field
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class GradeState(BaseModel):
    binary_score: str=Field(description="Relevance score 'yes' or 'no'")